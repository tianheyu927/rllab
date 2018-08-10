from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc.instrument import stub, run_experiment_lite
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place_mil import SawyerPickPlaceMILEnv
from multiworld.core.flat_goal_env import FlatGoalEnv

# xml_file = '/home/kevin/multiworld/multiworld/envs/assets/sawyer_xyz/sawyer_pick_and_place_vase1.xml'
xml_file = '/home/kevin/multiworld/multiworld/envs/assets/sawyer_xyz/sawyer_pick_and_place_vase1_distr.xml'

def run_task(*_):

    env = TfEnv(FlatGoalEnv(SawyerPickPlaceMILEnv(**{'xml_file': xml_file, 'include_distractors': True, 'include_goal': True})))
    policy = GaussianMLPPolicy(
        # name="policy",
        name="policy",
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(128, 128, 128)
        # hidden_sizes=(32, 32)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)
    # import tensorflow as tf
    # import joblib
    # with tf.Session() as sess:
    #     data = joblib.load('data/local/trpo-sawyer-pick-place/trpo_sawyer_pick_place_2018_08_07_02_33_41_0001/itr_800.pkl')
    
    algo = TRPO(
        env=env, #data['env'],
        policy=policy, #data['policy'],
        baseline=baseline, #data['baseline'],
        batch_size=100*500,
        max_path_length=200, #120,#130, #130,
        n_itr=1000, #1000
        discount=0.99,#0.99,
        step_size=0.01,#0.01,
        force_batch_sampler=True,
        # optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
    )
    algo.train()

run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=6,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="gap",
    snapshot_gap=25,
    exp_prefix='trpo_sawyer_pick_place',
    # resume_from='data/local/trpo-sawyer-pick-place/trpo_sawyer_pick_place_2018_08_07_02_33_41_0001/itr_800.pkl',
    python_command='python3',
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    plot=True,
)