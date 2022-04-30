using LinearAlgebra
using Test
using ReinforcementLearningBase.RLBase
using ReinforcementLearning

include("../Environment/SpacecraftEnv_GT.jl")
#using .SpacecraftEnv_GT

run(
    RandomPolicy(),
    SpacecraftEnv_GT.SpacecraftEnv(),
    StopAfterStep(1000),
    TotalRewardPerEpisode()
)