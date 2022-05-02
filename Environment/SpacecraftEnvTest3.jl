using LinearAlgebra
using Test
using ReinforcementLearningBase.RLBase
using Random
using Random: AbstractRNG
using ClosedIntervals
using ReinforcementLearningZoo
using ReinforcementLearningBase
using ReinforcementLearningCore
using ReinforcementLearningEnvironments
using StableRNGs
using CommonRLInterface
using Flux
using CommonRLInterface.Wrappers: QuickWrapper
using VegaLite
using DataFrames: DataFrame
using POMDPs: isterminal
using QuickPOMDPs: QuickPOMDP
using POMDPModelTools: Deterministic, Uniform, SparseCat
using POMDPPolicies: FunctionPolicy
using POMDPSimulators: RolloutSimulator
using Statistics: mean
using BeliefUpdaters: DiscreteUpdater 
import POMDPs
using Setfield
include("DQN.jl")
include("DDPGPolicy.jl")
include("Common.jl")

export SpacecraftEnv

struct SpacecraftEnvParams{T}
    min_radius::T
    max_radius::T
    goal_distance::T
    Isp::T
    g0::T
    Thrust::T
    mu::T
    max_steps::Int
    timestep::Int
end

Base.show(io::IO, params::SpacecraftEnvParams) = print(
    io,
    join(["$p=$(getfield(params, p))" for p in fieldnames(SpacecraftEnvParams)], ","),
)

function SpacecraftEnvParams(;
    T = Float64,
    min_radius = 6571, #km (200 km from Earth surface)
    max_radius = 7571, #km (1000 km above Earth surface)
    goal_distance = 50, #km (goal distance between sc0 and target spacecrafts)
    # goal_velocity = 7.3043, #km/s
    Isp = 300, #hydrogen engines
    g0 = 0.00981, #km/s^2
    Thrust = 20, #N
    mu = 398600.4, #km^3/s^2
    max_steps = 20000,
    timestep = 1,
)
    
    SpacecraftEnvParams{T}(
        min_radius,
        max_radius,
        goal_distance,
        # goal_velocity,
        Isp,
        g0,
        Thrust,
        mu,
        max_steps,
        timestep,
    )
end

mutable struct SpacecraftEnv{A,T,ACT,R<:AbstractRNG} <: AbstractEnv
    params::SpacecraftEnvParams{T}
    action_space::A
    observation_space::Space{Vector{ClosedInterval{T}}}
    state::Vector{T}
    action::ACT
    reward::Float64
    done::Bool
    t::Int
    rng::R
end

function SpacecraftEnv(;
    T = Float64,
    rng = Random.GLOBAL_RNG,
    kwargs...,
)
    
    params = SpacecraftEnvParams(; T = T, kwargs...)

    action_space = Base.OneTo(5);
    env = SpacecraftEnv(
        params,
        action_space,
        Space([6000..10000.0,0.0..1000.0,-100.0..100.0,0.0..100.0,0.0..1100.0,-100.0..100.0,6000..10000.0,0.0..1000.0,-100.0..100.0,0.0..100.0,0.0..1100.0,-100.0..100.0]),
        zeros(T, 12),
        rand(action_space),
        0.0,
        false,
        0,
        rng,
    )
    reset!(env)
    env
end

Random.seed!(env::SpacecraftEnv, seed) = Random.seed!(env.rng, seed)
RLBase.action_space(env::SpacecraftEnv) = env.action_space
RLBase.state_space(env::SpacecraftEnv) = env.observation_space
RLBase.reward(env::SpacecraftEnv{A,T}) where {A,T} = SpacecraftEnvReward(env)
RLBase.is_terminated(env::SpacecraftEnv) = env.done
RLBase.state(env::SpacecraftEnv) = env.state

function RLBase.reset!(env::SpacecraftEnv{A,T}) where {A,T}
    # Servicing SC
    env.state[1] = 6571 #km
    env.state[2] = rand(env.rng, T) #rad
    env.state[3] = 0.0 #km/s
    env.state[4] = 7.7885 #km/s
    env.state[5] = 1000.0 #kg
    env.state[6] = 0 #kg/s

    # Space Station - to be refuelled
    env.state[7] = 6771 #km
    env.state[8] = 4.7125 #rad
    env.state[9] = 0.0 #km/s
    env.state[10] = 7.6726 #km/s
    env.state[11] = 1000.0 #kg
    env.state[12] = 0 #kg/s
    env.done = false
    env.t = 0
    nothing
end

##not sure what the next two funcs are for
# function (env::SpacecraftEnv{<:ClosedInterval})(a::AbstractFloat)
#     @assert a in env.action_space
#     env.action = a
#     _step!(env, a)
# end

function (env::SpacecraftEnv{<:Base.OneTo{Int}})(a::Int)
    @assert a in env.action_space
    env.action = a
    _step!(env, a-3)
end

function _step!(env::SpacecraftEnv)
    μ = env.params.mu
    sc = env.state
    throttle = env.action
    Thr = env.params.Thrust*throttle
    timestep = env.params.timestep
    env.t += timestep
    

    r,θ,vr,vθ,m,m_dot = sc[1:6]
    r_dot, = vr
    θ_dot = vθ/r 
    vr_dot = vθ^2/r - μ/(r^2)
    vθ_dot = -vr*vθ/r + Thr/(m-m_dot*timestep) #should probably change this, as this doesn't mean only in dom. Even though optimally it isn't important
    r_new = r+r_dot*timestep
    θ_new = θ+θ_dot*timestep
    vr_new = vr+vr_dot*timestep
    vθ_new = vθ+vθ_dot*timestep
    m_dot = 0
    m_dot = -abs(Thr)/(env.params.Isp*env.params.g0)
    m_new = m+m_dot*timestep
    env.state[1:6] = [r_new,θ_new,vr_new,vθ_new,m_new,m_dot]

    r,θ,vr,vθ,m,m_dot = sc[7:12]
    r_dot, = vr
    θ_dot = vθ/r 
    vr_dot = vθ^2/r - μ/(r^2)
    vθ_dot = -vr*vθ/r
    r_new = r+r_dot*timestep
    θ_new = θ+θ_dot*timestep
    vr_new = vr+vr_dot*timestep
    vθ_new = vθ+vθ_dot*timestep
    m_dot = 0
    env.state[7:12] = [r_new,θ_new,vr_new,vθ_new,m_new,m_dot]

    env.done = 
        env.t>=env.params.max_steps ||
        env.state[5] <= 0.0 #out of fuel
    
    r = env.done ? SpacecraftFinalRewardTest(env) : SpacecraftEnvReward(env)
    return r
    # nothing
end

function SpacecraftEnvReward(env::SpacecraftEnv)
    state = env.state
    r0 = state[1]
    θ0 = state[2]
    r = state[7]
    θ = state[8]
    distance = sqrt((r*cos(θ) - r0*cos(θ0))^2 + (r*sin(θ) - r0*sin(θ0))^2)
    if distance < env.params.goal_distance
        return(100.0)
    else
        return(0.0)
    end
end

function SpacecraftFinalRewardTest(env::SpacecraftEnv)
    state = env.state
    r0 = state[1]
    θ0 = state[2]
    r = state[7]
    θ = state[8]
    distance = sqrt((r*cos(θ) - r0*cos(θ0))^2 + (r*sin(θ) - r0*sin(θ0))^2)
    return(-distance)
end

using ReinforcementLearning

# @testset "ReinforcementLearningZoo.jl" begin
#     include("TabularRandomPolicy.jl")
# end


env = SpacecraftEnv()
# RLBase.test_runnable!(env)
run(DQN(env),env,StopAfterStep(1000),TotalRewardPerEpisode())
# run(,env,StopAfterStep(1000),TotalRewardPerEpisode())
