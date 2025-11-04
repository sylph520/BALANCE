import os
import glob
import copy
import shutil
import logging
import pickle
from gym_db.common import EnvironmentType
from balance.experiment import Experiment
import argparse
import datetime
import tensorflow as tf
import numpy as np

import random

import gym_db  # noqa: F401
from index_selection_evaluation.selection.workload import Workload

use_gpu = os.environ['CUDA_VISIBLE_DEVICES']

import math
import datetime
from stable_baselines.common.callbacks import BaseCallback

class PPODiagnosticsCallback(BaseCallback):
    """
    Logs and warns when PPO training shows unhealthy patterns:
    - critic collapse (explained variance < 0)
    - entropy too high/low (relative entropy out of [0.05, 0.2])
    - frozen updates (KL < 0.001 or clipfrac < 0.05)
    Adds text summaries to TensorBoard as well.
    """

    def __init__(self, action_space_size, verbose=1):
        super(PPODiagnosticsCallback, self).__init__(verbose)
        self.A = float(action_space_size)
        self.warning_log = []
        self.logger_ref = None  # Will be set during training

    def _init_callback(self):
        # Called once training starts — model.logger now exists
        self.logger_ref = getattr(self.model, "logger", None)

    def _record_warning(self, message):
        """Add warning to rolling log + TensorBoard."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{timestamp}] {message}"
        self.warning_log.append(full_msg)
        self.warning_log = self.warning_log[-15:]  # keep last 15 warnings

        # Print to console
        if self.verbose > 0:
            print(f"\033[93m⚠️ {full_msg}\033[0m")

        # Log to TensorBoard if available
        if self.logger_ref is not None:
            tb_text = "\n".join(self.warning_log)
            self.logger_ref.record("diagnostics/warnings_text", tb_text)

    def _on_step(self) -> bool:
        if self.logger_ref is None:
            return True  # Skip until logger is ready

        logs = getattr(self.logger_ref, "name_to_value", {}) or {}

        ev = logs.get("train/explained_variance")
        ent = logs.get("loss/entropy_loss")
        kl = logs.get("loss/approximate_kullback-leibler")
        clipfrac = logs.get("loss/clip_factor")

        # ---- Critic health ----
        if ev is not None and ev < 0:
            self._record_warning(f"Critic collapse detected (EV={ev:.3f})")

        # ---- Entropy checks ----
        if ent is not None:
            rel_ent = ent / math.log(self.A)
            self.logger_ref.record("diagnostics/relative_entropy", rel_ent)
            if rel_ent > 0.3:
                self._record_warning(f"Entropy too high (rel={rel_ent:.2f}) — reduce ent_coef")
            elif rel_ent < 0.03:
                self._record_warning(f"Entropy too low (rel={rel_ent:.2f}) — exploration dying")

        # ---- PPO update health ----
        if kl is not None and kl < 0.001:
            self._record_warning(f"PPO nearly frozen (KL={kl:.4f})")

        if clipfrac is not None and clipfrac < 0.05:
            self._record_warning(f"Updates too conservative (clipfrac={clipfrac:.3f})")

        return True

class VecNormDiagCallback(BaseCallback):
    """
    Print VecNormalize stats every few updates.
    """
    def __init__(self, check_freq=5000, verbose=0):
        super(VecNormDiagCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self):
        # 'self.training_env' is VecNormalize(env, ...)
        if self.num_timesteps % self.check_freq == 0:
            try:
                vec = self.training_env
                print(f"\n[Diag] step={self.num_timesteps}")
                print("Reward RMS mean:", vec.ret_rms.mean)
                print("Reward RMS var :", vec.ret_rms.var)
                print("Obs RMS mean:", vec.obs_rms.mean.mean())
                print("Obs RMS var :", vec.obs_rms.var.mean())
            except Exception as e:
                print("[Diag] Could not read VecNormalize stats:", e)
        return True

class EntropySchedulerCallback(BaseCallback):
    """
    CORRECTED: A custom callback that linearly decays the entropy coefficient (ent_coef)
    by directly accessing the model.ent_coef attribute, which is correct for PPO2 in SB2.
    """
    def __init__(self, start_value, end_value, total_timesteps, verbose=0):
        super(EntropySchedulerCallback, self).__init__(verbose)
        self.start_value = start_value
        self.end_value = end_value
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        # Calculate the fraction of total training completed
        progress = self.num_timesteps / self.total_timesteps
        
        # Linear decay formula
        new_ent_coef = self.start_value - (self.start_value - self.end_value) * progress
        
        # **CORRECTED ACTION:** Set the new entropy coefficient by modifying the attribute directly
        self.model.ent_coef = new_ent_coef

        return True # Continue training

def _get_latest_tb_run_id(log_path, log_name):
    if not log_path or not log_name:
        return 0
    pattern = os.path.join(log_path, f"{log_name}_*")
    max_run_id = 0
    for path in glob.glob(pattern):
        file_name = os.path.basename(path)
        parts = file_name.rsplit("_", 1)
        if len(parts) != 2:
            continue
        prefix, suffix = parts
        if prefix != log_name or not suffix.isdigit():
            continue
        run_id = int(suffix)
        if run_id > max_run_id:
            max_run_id = run_id
    return max_run_id


def _resolve_tb_run_dir(log_path, log_name, new_tb_log):
    if not log_path or not log_name:
        return None
    latest_run_id = _get_latest_tb_run_id(log_path, log_name)
    run_id = latest_run_id + 1 if new_tb_log else latest_run_id
    if run_id < 0:
        run_id = 0
    return os.path.join(log_path, f"{log_name}_{run_id}")


def run_single_experiment(configuration_file, test_only, ts=16000, uni_freq=False, weight_path='',
                          fix_index_count=0, dmx_sz=0,
                          test_workload_from_file='', test_workload_qids='', newf=False,
                          input_workload: Workload=None,
                          random_seed=0, shuffle=False, debug_print=False,
                          cli_disable_precedent_masking=None, cli_enable_precedent_masking=None,
                          tb_log_path='',
                        #   lr=0.00025, ec=.01, cr=0.2, ns=128, gamma=0.99,
                          lr=-1, ec=-1, cr=-1, ns=-1, gamma=-1,
                          dump_initial_config=True, num_parallel_env=-1,
                          model_path='',
                          reward_scale=1.0):
    CONFIGURATION_FILE = configuration_file
    if tb_log_path  == 'None':
        tb_log_path = None
    else:
        tb_log_path = tb_log_path
    np.random.seed(random_seed)
    random.seed(random_seed)

    if weight_path:
        uni_freq = False
    else:
        uni_freq = True

    logging.warning("use gpu:" + use_gpu)
    if test_only:
        experiment = Experiment(CONFIGURATION_FILE, skip_folder_creation=True, uni_freq=uni_freq, fix_index_count=fix_index_count, ts=ts,
                newf=newf, random_seed=random_seed, debug_print=debug_print, dmx_sz=dmx_sz,
                cli_disable_precedent_masking=cli_disable_precedent_masking, cli_enable_precedent_masking=cli_enable_precedent_masking,
                lr=lr, ec=ec, cr=cr, ns=ns, gamma=gamma)
        from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize

        experiment.prepare(input_workload, weight_path=weight_path, shuffle=shuffle, input_workload_path=test_workload_from_file)
        if input_workload:
            input_qids = [q.nr  for q in input_workload.queries]
        else:
            input_qids = [q.nr  for q in experiment.workload_generator.wl_testing[0][0].queries]

        experiment_base_name = experiment.id
        folder_path = experiment.experiment_folder_path

        if os.path.exists(folder_path):
            tb_run_dir = None
            tb_base_path = tb_log_path if tb_log_path else "tensor_log"
            tb_log_name = 'tblog_' + folder_path.strip('/').split('/')[-1]
            if tb_base_path:
                latest_run_id = _get_latest_tb_run_id(tb_base_path, tb_log_name)
                candidate_tb_dir = os.path.join(tb_base_path, f"{tb_log_name}_{latest_run_id}")
                if os.path.isdir(candidate_tb_dir):
                    tb_run_dir = candidate_tb_dir

            candidate_model_paths = []
            if model_path:
                candidate_model_paths.append(model_path)
            if tb_run_dir:
                candidate_model_paths.append(os.path.join(tb_run_dir, "final_model.zip"))
            candidate_model_paths.append(os.path.join(folder_path, "final_model.zip"))

            resolved_model_path = next((p for p in candidate_model_paths if p and os.path.exists(p)), None)

            if resolved_model_path:
                logging.info(f"Loading model from: {resolved_model_path}")
                if experiment.config["rl_algorithm"]["stable_baselines_version"] == 2:
                    from stable_baselines.ppo2 import ppo2
                    algorithm_class = ppo2.PPO2
                    experiment.model_type = algorithm_class
                    experiment.source_model_type = algorithm_class
                else:
                    raise ValueError

                experiment.start_learning()
                model = experiment.model_type.load(resolved_model_path)
                experiment.set_model(model)

                if test_workload_from_file:
                    logging.info(f"Using custom test workload from file: {test_workload_from_file}")
                    query_ids = None
                    if test_workload_qids:
                        try:
                            query_ids = [int(qid.strip()) for qid in test_workload_qids.split(',')]
                        except ValueError:
                            logging.error("Invalid format for --test_workload_qids. Please provide a comma-separated list of integers.")
                            exit(1)
                    test_wl = experiment.workload_from_sql_file(test_workload_from_file, selection_qids=query_ids)
                    if fix_index_count > 0:
                        test_wl.budget = fix_index_count
                    test_env_dummy = DummyVecEnv([experiment.make_env(0, EnvironmentType.TESTING, workloads_in=[test_wl])])
                else:
                    custom_wl = True
                    if custom_wl:
                        logging.info("Using custom hardcoded test workload.")
                        if input_workload:
                            test_wl = input_workload
                        else:
                            if not uni_freq:
                                test_wl = experiment.workload_generator._workloads_from_tuples([tuple((list(range(1, 21)), [1]*20))])[0]
                            else:
                                test_wl = experiment.workload_generator._workloads_from_tuples([tuple((input_qids, [1]*20))])[0]
                        test_wl.budget = fix_index_count
                        test_env_dummy = DummyVecEnv([experiment.make_env(0, EnvironmentType.TESTING, workloads_in=[test_wl], reward_scale=reward_scale)])
                    else:
                        logging.info("Using default test workload from configuration.")
                        test_env_dummy = DummyVecEnv([experiment.make_env(0, EnvironmentType.TESTING)])

                vec_candidates = []
                if tb_run_dir:
                    vec_candidates.append(os.path.join(tb_run_dir, "vec_normalize.pkl"))
                vec_candidates.append(os.path.join(folder_path, "vec_normalize.pkl"))

                vec_norm_path = next((p for p in vec_candidates if os.path.exists(p)), None)

                if vec_norm_path:
                    logging.info(f"Loading normalization statistics from: {vec_norm_path}")
                    test_env = VecNormalize.load(vec_norm_path, test_env_dummy)
                    test_env.training = False
                    test_env.norm_reward = False
                else:
                    logging.warning("Could not find normalization statistics. Using new VecNormalize wrapper.")
                    test_env = VecNormalize(
                        test_env_dummy,
                        norm_obs=True,
                        norm_reward=False,
                        gamma=experiment.config["rl_algorithm"]["gamma"],
                        training=False,
                    )
                model.set_env(test_env)
                rec_start_time = datetime.datetime.now()
                training_env = model.get_vec_normalize_env()
                if training_env is not None:
                    experiment.sync_envs_normalization(training_env, test_env)

                n_eval_episodes = experiment.config["workload"]["validation_testing"]["number_of_workloads"]
                episode_performances = experiment._evaluate_model(model, test_env, n_eval_episodes)
                rec_end_time = datetime.datetime.now()
                logging.info(f"Evaluation completed. Performance: {episode_performances}")
                print(f"Final Mean IR: {100 - episode_performances[1]:.2f}")
                experiment.training_end_time = datetime.datetime.now()
                print(f"rectime is {(rec_end_time - rec_start_time).total_seconds() * 1000}")
                # experiment.finishmy()
                return experiment.experiment_folder_path
                # exit(0)
            else:
                logging.warning("No saved model artifacts found. Checked paths: %s", candidate_model_paths)
                raise ValueError
        else:
            logging.warning(f"No experiment folders found for {experiment_base_name}, proceeding with training")
            raise ValueError
    else:
        experiment = Experiment(CONFIGURATION_FILE, uni_freq=uni_freq, fix_index_count=fix_index_count, ts=ts, dmx_sz=dmx_sz,
                    random_seed=random_seed, debug_print=debug_print, cli_disable_precedent_masking=cli_disable_precedent_masking,
                    cli_enable_precedent_masking=cli_enable_precedent_masking,
                    skip_folder_creation=True,
                    lr=lr, ec=ec, cr=cr, ns=ns, gamma=gamma, num_parallel_env=num_parallel_env)

        if experiment.config["rl_algorithm"]["stable_baselines_version"] == 2:
            from stable_baselines.common.callbacks import EvalCallbackWithTBRunningAverage
            from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
            from stable_baselines.ppo2 import ppo2, ppo2_BALANCE
            source_algorithm_class = ppo2.PPO2
            if experiment.config['rl_algorithm']['algorithm'] == 'PPO2':
                algorithm_class = ppo2.PPO2
            elif experiment.config['rl_algorithm']['algorithm'] == 'ppo2_BALANCE':
                algorithm_class = ppo2_BALANCE.PPO2
            else:
                raise ValueError(f"unknown algorithm type {experiment.config['rl_algorithm']['algorihtm']}")
        else:
            raise ValueError

        experiment.prepare(input_workload, weight_path=weight_path, shuffle=shuffle)
        with open(f"{experiment.experiment_folder_path}/experiment_object.pickle", "wb") as handle:
            pickle.dump(experiment, handle, protocol=pickle.HIGHEST_PROTOCOL)
        ParallelEnv = SubprocVecEnv if experiment.config["parallel_environments"] > 1 else DummyVecEnv

        training_env = ParallelEnv(
            [experiment.make_env(env_id) for env_id in range(experiment.config["parallel_environments"])]
        )
        training_env = VecNormalize(
            training_env, gamma=experiment.gamma, training=True,
            norm_obs=True, norm_reward=True,
            clip_obs=10., clip_reward=10.
        )

        temac = []

        experiment.source_model_type = source_algorithm_class
        experiment.model_type = algorithm_class

        if len(experiment.model_pool) > 0:
            logging.info(f"Policy transfer enabled. Loading {len(experiment.model_pool)} source models.")
            for model_path in experiment.model_pool:
                temac.append(experiment.source_model_type.load(model_path))

            model: ppo2_BALANCE.PPO2 = algorithm_class(
                policy=experiment.config["rl_algorithm"]["policy"],
                env=training_env,
                verbose=2,
                seed=experiment.config["random_seed"],
                gamma=experiment.gamma,
                tensorboard_log=tb_log_path,
                acc=temac,
                policy_kwargs=copy.copy(
                    experiment.config["rl_algorithm"]["model_architecture"]
                ),  # This is necessary because SB modifies the passed dict.
                **experiment.config["rl_algorithm"]["args"],
            )
        else:
            model: ppo2.PPO2 = ppo2.PPO2(
                policy=experiment.config["rl_algorithm"]["policy"],
                env=training_env,
                verbose=2,
                seed=experiment.config["random_seed"],
                gamma=experiment.gamma,
                tensorboard_log=tb_log_path,
                policy_kwargs=copy.copy(
                    experiment.config["rl_algorithm"]["model_architecture"]
                ),  # This is necessary because SB modifies the passed dict.
                **experiment.config["rl_algorithm"]["args"],
                )
        logging.warning(f"Creating model with NN architecture: {experiment.config['rl_algorithm']['model_architecture']}")

        experiment.set_model(model)
        if dump_initial_config:
            experiment.dump_config_snapshot(experiment.experiment_folder_path, filename="config.initial.json")
        tb_run_dir = None

        callback_test_env = VecNormalize(
            DummyVecEnv([experiment.make_env(0, EnvironmentType.TESTING)]),
            norm_obs=True, norm_reward=False,
            gamma=experiment.config["rl_algorithm"]["gamma"],
            training=False,
        )
        test_callback = EvalCallbackWithTBRunningAverage(
            n_eval_episodes=experiment.config["workload"]["validation_testing"]["number_of_workloads"],
            eval_freq=round(experiment.config["validation_frequency"] / experiment.config["parallel_environments"]),
            eval_env=callback_test_env,
            verbose=1, name="test",
            deterministic=True,
            comparison_performances=experiment.comparison_performances["test"],
        )

        callback_validation_env = VecNormalize(
            DummyVecEnv([experiment.make_env(0, EnvironmentType.VALIDATION)]),
            norm_obs=True, norm_reward=False,
            gamma=experiment.config["rl_algorithm"]["gamma"],
            training=False,
        )
        validation_callback = EvalCallbackWithTBRunningAverage(
            n_eval_episodes=experiment.config["workload"]["validation_testing"]["number_of_workloads"],
            eval_freq=round(experiment.config["validation_frequency"] / experiment.config["parallel_environments"]),
            eval_env=callback_validation_env,
            best_model_save_path=experiment.experiment_folder_path,
            verbose=1, name="validation",
            deterministic=True,
            comparison_performances=experiment.comparison_performances["validation"],
        )
        diag_callback = PPODiagnosticsCallback(action_space_size=training_env.action_space.n)
        vecnormCallback = VecNormDiagCallback(check_freq=5000)

        ent_callback = EntropySchedulerCallback(
            start_value=0.001, 
            end_value=0.00001, 
            total_timesteps=experiment.config['timesteps'],
        )
        # callbacks = [validation_callback, test_callback, diag_callback, ent_callback]
        callbacks = [validation_callback, test_callback, vecnormCallback]

        if len(experiment.multi_validation_wl) > 0:
            callback_multi_validation_env = VecNormalize(
                DummyVecEnv([experiment.make_env(0, EnvironmentType.VALIDATION, experiment.multi_validation_wl)]),
                norm_obs=True, norm_reward=False,
                gamma=experiment.config["rl_algorithm"]["gamma"],
                training=False,
            )
            multi_validation_callback = EvalCallbackWithTBRunningAverage(
                n_eval_episodes=len(experiment.multi_validation_wl),
                eval_freq=round(experiment.config["validation_frequency"] / experiment.config["parallel_environments"]),
                eval_env=callback_multi_validation_env,
                best_model_save_path=experiment.experiment_folder_path,
                verbose=1, name="multi_validation",
                deterministic=True,
                comparison_performances={},
            )
            callbacks.append(multi_validation_callback)

        experiment.start_learning()
        tb_log_name = 'tblog_'+experiment.experiment_folder_path.split('/')[-1]
        new_tb_log = model.num_timesteps == 0
        if tb_log_path:
            tb_run_dir = _resolve_tb_run_dir(tb_log_path, tb_log_name, new_tb_log)

        model.learn(
            total_timesteps=experiment.config["timesteps"],
            callback=callbacks,
            tb_log_name=tb_log_name, ids=experiment.config["id"]
        )
        experiment.finish_learning(
            training_env,
            validation_callback.moving_average_step * experiment.config["parallel_environments"],
            validation_callback.best_model_step * experiment.config["parallel_environments"],
        )
        if tb_run_dir:
            os.makedirs(tb_run_dir, exist_ok=True)
            model_artifacts = [
                fname
                for fname in os.listdir(experiment.experiment_folder_path)
                if fname.endswith(".zip")
            ]
            for artifact in model_artifacts:
                src = os.path.join(experiment.experiment_folder_path, artifact)
                dst = os.path.join(tb_run_dir, artifact)
                try:
                    shutil.copy2(src, dst)
                except Exception as exc:
                    logging.warning("Failed to copy model artifact %s to TensorBoard dir %s: %s", artifact, tb_run_dir, exc)
            vec_normalize_path = os.path.join(experiment.experiment_folder_path, "vec_normalize.pkl")
            if os.path.exists(vec_normalize_path):
                try:
                    shutil.copy2(vec_normalize_path, os.path.join(tb_run_dir, "vec_normalize.pkl"))
                except Exception as exc:
                    logging.warning("Failed to copy vec_normalize.pkl to TensorBoard dir %s: %s", tb_run_dir, exc)
            if dump_initial_config:
                initial_src = os.path.join(experiment.experiment_folder_path, "config.initial.json")
                if os.path.exists(initial_src):
                    try:
                        shutil.copy2(initial_src, os.path.join(tb_run_dir, "config.initial.json"))
                    except Exception as exc:
                        logging.warning("Failed to copy initial config to TensorBoard dir %s: %s", tb_run_dir, exc)
            experiment.dump_config_snapshot(tb_run_dir)

        with open(f"{experiment.experiment_folder_path}/workload_dic.pickle", "wb") as handle:
            pickle.dump([training_env.get_attr("dic")[0], callbacks[0].eval_env.get_attr("dic")[0], callbacks[1].eval_env.get_attr("dic")[0]], handle, protocol=pickle.HIGHEST_PROTOCOL)

        experiment.finishmy()

        return experiment.experiment_folder_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wk_type', type=str, default='tpch')
    parser.add_argument('--config', type=str, help='Path to configuration file (overrides wk_type)')
    parser.add_argument('--ts', type=int, help='the number of time steps to train', default=0)
    parser.add_argument('--load_model', type=str, help='Path to saved model to test instead of training')
    parser.add_argument('--test_only', action='store_true', help='Load and test latest model from config experiment folder')
    parser.add_argument('--uni_freq', action='store_true', default=True)
    # parser.add_argument('--weight_path', type=str, default='query_files/tpch12/weight1.pkl')
    # parser.add_argument('--weight_path', type=str, default='query_files/TPCHC/varied_weights.pkl')
    parser.add_argument('--weight_path', type=str, default='')
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--fix_index_count', type=int, default=0)
    parser.add_argument('--test_workload_file', type=str, help='Path to a .sql file to use as a custom test workload.')
    parser.add_argument('--test_workload_qids', type=str, help='Comma-separated list of query IDs for the custom test workload.')
    parser.add_argument('--newf', action='store_true', default=False)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--debug_print', action='store_true', help='Enable debug print statements')
    parser.add_argument('--disable_precedent_masking', action='store_const', const=True, default=None, help='Disable precedent masking')
    parser.add_argument('--enable-precedent-masking', action='store_const', const=False, default=None, help='Enable precedent masking')
    parser.add_argument('--dmx_sz', type=int, default=0)
    parser.add_argument('--tb_log', type=str, default='tensor_log')
    parser.add_argument('--skip_initial_config_dump', action='store_true', help='Skip writing pre-training config snapshots.')
    parser.add_argument('--num_parallel_env', type=int, default=-1, help='Overwrite the number of parallel environments in the config file.')
    parser.add_argument('--lr', type=float, default=-1)
    parser.add_argument('--ns', type=int, default=-1)
    parser.add_argument('--ec', type=float, default=-1)
    parser.add_argument('--cr', type=float, default=-1)
    parser.add_argument('--gamma', type=float, default=-1)
    parser.add_argument('--reward_scale', type=float, default=1.0)
    args = parser.parse_args()

    if args.config:
        config_file = args.config
    else:
        config_file = f"experiments/{(args.wk_type).lower()}_conf/{(args.wk_type).lower()}.json"

    if args.weight_path:
        uni_freq_flag = False
    else:
        uni_freq_flag=args.uni_freq

    run_single_experiment(config_file, test_only=args.test_only, uni_freq=uni_freq_flag, weight_path=args.weight_path,\
                            fix_index_count=args.fix_index_count, ts=args.ts,
                            test_workload_from_file=args.test_workload_file, test_workload_qids = args.test_workload_qids,
                            dmx_sz = args.dmx_sz,
                            newf=args.newf, shuffle=args.shuffle, debug_print=args.debug_print,
                            cli_disable_precedent_masking=args.disable_precedent_masking,
                            cli_enable_precedent_masking=args.enable_precedent_masking,
                            tb_log_path = args.tb_log,
                            lr=args.lr, ec=args.ec, cr=args.cr, ns=args.ns, gamma=args.gamma,
                            dump_initial_config=not args.skip_initial_config_dump,
                            num_parallel_env=args.num_parallel_env,
                            reward_scale=args.reward_scale,
                            model_path=args.load_model)
