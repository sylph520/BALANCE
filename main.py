import copy
import importlib
import logging
import pickle
import sys
import gym_db  # noqa: F401
from gym_db.common import EnvironmentType
from balance.experiment import Experiment
import argparse
import os

# For full determinism, set hash seed and seed random libraries at the start.
os.environ['PYTHONHASHSEED'] = '0'
import numpy as np
np.random.seed(0)
import random
random.seed(0)


use_gpu = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--wk_type', type=str, default='tpch')
    parser.add_argument('--config', type=str, help='Path to configuration file (overrides wk_type)')
    parser.add_argument('--ts', type=int, help='the number of time steps to train', default=0)
    parser.add_argument('--load_model', type=str, help='Path to saved model to test instead of training')
    parser.add_argument('--test_only', action='store_true', help='Load and test latest model from config experiment folder')
    parser.add_argument('--uni_freq', action='store_true', default=False)
    parser.add_argument('--fix_index_count', type=int, default=0)
    parser.add_argument('--test_workload', type=str, help='Path to a .sql file to use as a custom test workload.')
    parser.add_argument('--test_workload_qids', type=str, help='Comma-separated list of query IDs for the custom test workload.')
    args = parser.parse_args()

    if args.config:
        CONFIGURATION_FILE = args.config
    else:
        CONFIGURATION_FILE = f"experiments/{(args.wk_type).lower()}.json"
    # CONFIGURATION_FILE = "experiments/tpcds.json"

    logging.warning("use gpu:" + use_gpu)
    # setup the experiment from configuration, random seed, and related method info
    # create or replace experiment result folder
    # If test_only flag is set, initialize experiment without folder deletion for testing
    # __import__('ipdb').set_trace()
    if args.test_only:
        experiment = Experiment(CONFIGURATION_FILE, skip_folder_creation=True, uni_freq=args.uni_freq, fix_index_count=args.fix_index_count, ts=args.ts)
        import os
        from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize

        # Prepare the experiment (this may be needed for environment setup)
        experiment.prepare()

        # Infer experiment folder name from config id
        experiment_base_name = experiment.id  # This comes from config["id"]

        # Look for the latest experiment folder with this name
        # folder_path = f"experiment_results/ID_{experiment_base_name}"
        folder_path = experiment.experiment_folder_path

        if os.path.exists(folder_path):
            # Get the most recent folder (by modification time)
            model_path = os.path.join(folder_path, "final_model.zip")

            if os.path.exists(model_path):
                logging.info(f"Loading model from: {model_path}")

                # Set up model_type before loading (this is normally done later in the code)
                if experiment.config["rl_algorithm"]["stable_baselines_version"] == 2:
                    from stable_baselines.ppo2 import ppo2, ppo2_BALANCE
                    algorithm_class = ppo2.PPO2
                    experiment.model_type = algorithm_class
                    experiment.source_model_type = algorithm_class
                else:
                    raise ValueError

                # Load the saved model
                model = experiment.model_type.load(model_path)

                if args.test_workload:
                    logging.info(f"Using custom test workload from file: {args.test_workload}")
                    query_ids = None
                    if args.test_workload_qids:
                        try:
                            query_ids = [int(qid.strip()) for qid in args.test_workload_qids.split(',')]
                        except ValueError:
                            logging.error("Invalid format for --test_workload_qids. Please provide a comma-separated list of integers.")
                            exit(1)
                    test_wl = experiment.workload_from_sql_file(args.test_workload, selection_qids=query_ids)
                    if args.fix_index_count > 0:
                        test_wl.budget = args.fix_index_count
                    test_env_dummy = DummyVecEnv([experiment.make_env(0, EnvironmentType.TESTING, workloads_in=[test_wl])])
                else:
                    # custom_wl = False
                    custom_wl = True
                    # Create test environment with default or custom workloads
                    if custom_wl:
                        logging.info("Using custom hardcoded test workload.")
                        if not args.uni_freq:
                            test_wl = experiment.workload_generator._workloads_from_tuples([tuple((list(range(1, 21)), [1]*20))])[0]
                        else:
                            test_wl = experiment.workload_generator._workloads_from_tuples([tuple(([20, 5, 17, 2, 4, 19, 3, 1, 10, 16, 15, 11, 12, 13, 18, 6, 14, 7, 9, 8], [1]*20))])[0]
                        test_wl.budget = 3
                        test_env_dummy = DummyVecEnv([experiment.make_env(0, EnvironmentType.TESTING, workloads_in=[test_wl])])
                    else:
                        logging.info("Using default test workload from configuration.")
                        test_env_dummy = DummyVecEnv([experiment.make_env(0, EnvironmentType.TESTING)])

                # Path to the normalization stats
                vec_norm_path = os.path.join(folder_path, "vec_normalize.pkl")

                # Load stats if they exist, otherwise create a new VecNormalize wrapper
                if os.path.exists(vec_norm_path):
                    logging.info(f"Loading normalization statistics from: {vec_norm_path}")
                    # Load the stats and wrap the dummy environment
                    test_env = VecNormalize.load(vec_norm_path, test_env_dummy)
                    test_env.training = False
                    test_env.norm_reward = False
                else:
                    logging.warning("Could not find normalization statistics. Using new VecNormalize wrapper.")
                    # If no saved stats, create a new VecNormalize wrapper
                    test_env = VecNormalize(
                        test_env_dummy,
                        norm_obs=True,
                        norm_reward=False,
                        gamma=experiment.config["rl_algorithm"]["gamma"],
                        training=False
                    )

                model.set_env(test_env)

                # Sync environments and evaluate (only if training_env exists)
                training_env = model.get_vec_normalize_env()
                if training_env is not None:
                    experiment.sync_envs_normalization(training_env, test_env)

                # Run evaluation
                n_eval_episodes = experiment.config["workload"]["validation_testing"]["number_of_workloads"]
                episode_performances = experiment._evaluate_model(model, test_env, n_eval_episodes)

                logging.info(f"Evaluation completed. Performance: {episode_performances}")
                print(f"Mean performance: {episode_performances[1]:.2f}")

                # __import__('ipdb').set_trace()
                # experiment.finishmy(args.test_only)
                # Exit after testing if only testing was requested
                exit(0)
            else:
                logging.warning(f"No saved model found at {model_path}, proceeding with training")
        else:
            logging.warning(f"No experiment folders found for {experiment_base_name}, proceeding with training")
    else:
        # Normal training mode
        experiment = Experiment(CONFIGURATION_FILE, uni_freq=args.uni_freq, fix_index_count=args.fix_index_count, ts=args.num_ts)

    if experiment.config["rl_algorithm"]["stable_baselines_version"] == 2:
        from stable_baselines.common.callbacks import EvalCallbackWithTBRunningAverage
        from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
        from stable_baselines.ppo2 import ppo2, ppo2_BALANCE
        # algorithm_class = ppo2_BALANCE.PPO2
        # source_algorithm_class = ppo2_BALANCE.PPO2
        algorithm_class = ppo2.PPO2
        source_algorithm_class = ppo2.PPO2
    else:
        raise ValueError

    # setup self.schema, self.workload_generator (for training, validation and testing)
    # experiment budgets (randomly selected from fixed lists), and self.embedder
    experiment.prepare()
    with open(f"{experiment.experiment_folder_path}/experiment_object.pickle", "wb") as handle:
        pickle.dump(experiment, handle, protocol=pickle.HIGHEST_PROTOCOL)
    ParallelEnv = SubprocVecEnv if experiment.config["parallel_environments"] > 1 else DummyVecEnv

    training_env = ParallelEnv(
        [experiment.make_env(env_id) for env_id in range(experiment.config["parallel_environments"])]
    )
    training_env = VecNormalize(
        training_env, norm_obs=True, norm_reward=True, gamma=experiment.config["rl_algorithm"]["gamma"], training=True
    )
    temac = []

    experiment.source_model_type = source_algorithm_class
    experiment.model_type = algorithm_class

    if len(experiment.model_pool) > 0:
        path1 = "./experiment_results/source"
        path2 = "./experiment_results/source"
        path3 = "./experiment_results/source"

        experiment.Smodel_1 = experiment.source_model_type.load(path1 + "/f_s1.zip")
        experiment.Smodel_1.training = False
        experiment.Smodel_2 = experiment.source_model_type.load(path2 + "/f_s2.zip")
        experiment.Smodel_2.training = False
        experiment.Smodel_3 = experiment.source_model_type.load(path3 + "/f_s3.zip")
        experiment.Smodel_3.training = False

        temac.append(experiment.Smodel_1)
        temac.append(experiment.Smodel_2)
        temac.append(experiment.Smodel_3)

        model: ppo2_BALANCE.PPO2 = algorithm_class(
            policy=experiment.config["rl_algorithm"]["policy"],
            env=training_env,
            verbose=2,
            seed=experiment.config["random_seed"],
            gamma=experiment.config["rl_algorithm"]["gamma"],
            tensorboard_log="tensor_log",
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
            gamma=experiment.config["rl_algorithm"]["gamma"],
            tensorboard_log="tensor_log",
            policy_kwargs=copy.copy(
                experiment.config["rl_algorithm"]["model_architecture"]
            ),  # This is necessary because SB modifies the passed dict.
            **experiment.config["rl_algorithm"]["args"],
            )
    logging.warning(f"Creating model with NN architecture: {experiment.config['rl_algorithm']['model_architecture']}")

    experiment.set_model(model)
    # experiment.compare()

    callback_test_env = VecNormalize(
        DummyVecEnv([experiment.make_env(0, EnvironmentType.TESTING)]),
        norm_obs=True,
        norm_reward=False,
        gamma=experiment.config["rl_algorithm"]["gamma"],
        training=False,
    )
    test_callback = EvalCallbackWithTBRunningAverage(
        n_eval_episodes=experiment.config["workload"]["validation_testing"]["number_of_workloads"],
        eval_freq=round(experiment.config["validation_frequency"] / experiment.config["parallel_environments"]),
        eval_env=callback_test_env,
        verbose=1,
        name="test",
        deterministic=True,
        comparison_performances=experiment.comparison_performances["test"],
    )

    callback_validation_env = VecNormalize(
        DummyVecEnv([experiment.make_env(0, EnvironmentType.VALIDATION)]),
        norm_obs=True,
        norm_reward=False,
        gamma=experiment.config["rl_algorithm"]["gamma"],
        training=False,
    )
    validation_callback = EvalCallbackWithTBRunningAverage(
        n_eval_episodes=experiment.config["workload"]["validation_testing"]["number_of_workloads"],
        eval_freq=round(experiment.config["validation_frequency"] / experiment.config["parallel_environments"]),
        eval_env=callback_validation_env,
        best_model_save_path=experiment.experiment_folder_path,
        verbose=1,
        name="validation",
        deterministic=True,
        comparison_performances=experiment.comparison_performances["validation"],
    )
    callbacks = [validation_callback, test_callback]

    if len(experiment.multi_validation_wl) > 0:
        callback_multi_validation_env = VecNormalize(
            DummyVecEnv([experiment.make_env(0, EnvironmentType.VALIDATION, experiment.multi_validation_wl)]),
            norm_obs=True,
            norm_reward=False,
            gamma=experiment.config["rl_algorithm"]["gamma"],
            training=False,
        )
        multi_validation_callback = EvalCallbackWithTBRunningAverage(
            n_eval_episodes=len(experiment.multi_validation_wl),
            eval_freq=round(experiment.config["validation_frequency"] / experiment.config["parallel_environments"]),
            eval_env=callback_multi_validation_env,
            best_model_save_path=experiment.experiment_folder_path,
            verbose=1,
            name="multi_validation",
            deterministic=True,
            comparison_performances={},
        )
        callbacks.append(multi_validation_callback)

    # setup learning timestamp to self.training_start_time
    experiment.start_learning()

    model.learn(
        total_timesteps=experiment.config["timesteps"],
        callback=callbacks,
        tb_log_name=experiment.experiment_folder_path, ids=experiment.config["id"]
    )
    experiment.finish_learning(
        training_env,
        validation_callback.moving_average_step * experiment.config["parallel_environments"],
        validation_callback.best_model_step * experiment.config["parallel_environments"],
    )

    with open(f"{experiment.experiment_folder_path}/workload_dic.pickle", "wb") as handle:
        pickle.dump([training_env.venv.envs[0].dic, callbacks[0].eval_env.venv.envs[0].dic, callbacks[1].eval_env.venv.envs[0].dic], handle, protocol=pickle.HIGHEST_PROTOCOL)
    experiment.finishmy()

    print()
