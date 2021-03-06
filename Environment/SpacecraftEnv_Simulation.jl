using LinearAlgebra
using Test
using ReinforcementLearningBase.RLBase
using Random
using Random: AbstractRNG
using ClosedIntervals
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
    max_steps = 50000,
    timestep = 10,
)
    
    SpacecraftEnvParams{T}(
        min_radius,
        max_radius,
        goal_distance,
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

    action_space = Base.OneTo(3)
    env = SpacecraftEnv(
        params,
        action_space,
        Space([6000.0..10000.0,0.0..1000.0,-100.0..100.0,0.0..100.0,0.0..1100.0,-100.0..100.0,6000.0..10000.0,0.0..1000.0,-100.0..100.0,0.0..100.0,0.0..1100.0,-100.0..100.0]),
        zeros(T, 18),
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
RLBase.reward(env::SpacecraftEnv{A,T}) where {A,T} = env.done ? SpacecraftFinalRewardTest(env) : SpacecraftEnvReward(env)
RLBase.is_terminated(env::SpacecraftEnv) = env.done
RLBase.state(env::SpacecraftEnv) = env.state

function RLBase.reset!(env::SpacecraftEnv{A,T}) where {A,T}
    # Servicing SC
    env.state[1] = 6571 #km
    env.state[2] = 2.5 #rad
    env.state[3] = 0.0 #km/s
    env.state[4] = 7.7885 #km/s
    env.state[5] = 1000.0 #kg
    env.state[6] = 0 #kg/s

    # Space debris to be avoided
    env.state[7] = 7000 #km
    env.state[8] = 1.7125 #rad
    env.state[9] = 0.0 #km/s
    env.state[10] = 7.5461 #km/s
    env.state[11] = 1000.0 #kg
    env.state[12] = 0 #kg/s

    # Space Station - to be refuelled
    env.state[13] = 7471 #km
    env.state[14] = 3.7125 #rad
    env.state[15] = 0.0 #km/s
    env.state[16] = 7.3043 #km/s
    env.state[17] = 10.0 #kg
    env.state[18] = 0 #kg/s
    env.done = false
    env.t = 0
    nothing
end

function (env::SpacecraftEnv{<:ClosedInterval})(a::AbstractFloat)
    @assert a in env.action_space
    env.action = a
    _step!(env, a)
end

function (env::SpacecraftEnv{<:Base.OneTo{Int}})(a::Int)
    @assert a in env.action_space
    env.action = a
    _step!(env, a-2)
end

function _step!(env::SpacecraftEnv, throttle)
    ?? = env.params.mu
    sc = env.state
    Thr = env.params.Thrust*throttle
    timestep = env.params.timestep
    env.t += timestep
    

    r,??,vr,v??,m,m_dot = sc[1:6]
    r_dot, = vr
    ??_dot = v??/r
    vr_dot = v??^2/r - ??/(r^2) 
    v??_dot = -vr*v??/r + Thr/(m-norm(m_dot)*timestep) 
    r_new = r+r_dot*timestep
    ??_new = ??+??_dot*timestep
    vr_new = vr+vr_dot*timestep
    v??_new = v??+v??_dot*timestep
    m_dot = 0
    m_dot = -abs(Thr)/(env.params.Isp*env.params.g0)
    m_new = m+m_dot*timestep
    env.state[1:6] = [r_new,??_new,vr_new,v??_new,m_new,m_dot]


    r,??,vr,v??,m,m_dot = sc[7:12]
    r_dot, = vr
    ??_dot = v??/r 
    vr_dot = v??^2/r - ??/(r^2)
    v??_dot = -vr*v??/r
    r_new = r+r_dot*timestep
    ??_new = ??+??_dot*timestep
    vr_new = vr+vr_dot*timestep
    v??_new = v??+v??_dot*timestep
    m_dot = 0
    env.state[7:12] = [r_new,??_new,vr_new,v??_new,m_new,m_dot]


    r,??,vr,v??,m,m_dot = sc[13:18]
    r_dot, = vr
    ??_dot = v??/r 
    vr_dot = v??^2/r - ??/(r^2)
    v??_dot = -vr*v??/r
    r_new = r+r_dot*timestep
    ??_new = ??+??_dot*timestep
    vr_new = vr+vr_dot*timestep
    v??_new = v??+v??_dot*timestep
    m_dot = 0
    env.state[13:18] = [r_new,??_new,vr_new,v??_new,m_new,m_dot]


    env.done = 
        env.t>=env.params.max_steps ||
        env.state[5] <= 0.0 ||#out of fuel 
        env.state[1] > env.state[13] # beyond SS orbit

    nothing
end

function SpacecraftEnvReward(env::SpacecraftEnv)
    state = env.state
    r0 = state[1]
    ??0 = state[2]
    r = state[7]
    ?? = state[8]
    r_iss = state[13]
    ??_iss = state[14]
    distance_obs = sqrt((r*cos(??) - r0*cos(??0))^2 + (r*sin(??) - r0*sin(??0))^2)
    distance_iss = sqrt((r_iss*cos(??_iss) - r0*cos(??0))^2 + (r_iss*sin(??_iss) - r0*sin(??0))^2)
    if distance_iss < env.params.goal_distance
        print("Goal reached")
        return(1000) #(100.0)
        env.done = true
    elseif distance_obs <env.params.goal_distance
        print("Obstacle reached")
        return(-100) 
        env.done = true
    else
        return(0.0)
    end
end

function SpacecraftFinalRewardTest(env::SpacecraftEnv)
    state = env.state
    r0 = state[1]
    ??0 = state[2]
    r = state[7]
    ?? = state[8]
    r_iss = state[13]
    ??_iss = state[14]
    distance_obs = sqrt((r*cos(??) - r0*cos(??0))^2 + (r*sin(??) - r0*sin(??0))^2)
    distance_iss = sqrt((r_iss*cos(??_iss) - r0*cos(??0))^2 + (r_iss*sin(??_iss) - r0*sin(??0))^2)
    if distance_iss < env.params.goal_distance
        print("Goal reached")
        return(1000)
        env.done = true
    elseif distance_obs <env.params.goal_distance
        print("Obstacle reached")
        return(-100) 
        env.done = true
    else
        return(-distance_iss)
    end
end

using ReinforcementLearning

#env = SpacecraftEnv()
#RLBase.test_runnable!(env)
#run(RandomPolicy(),env,StopAfterStep(1000),TotalRewardPerEpisode())


#=======================================================================================================================================#

using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:BasicDQN},
    ::Val{:SpacecraftEnv},
    ::Nothing;
    seed = 123,
)
    rng = StableRNG(seed)
    env = SpacecraftEnv(; T = Float64, max_steps = 5000, rng = rng)
    ns, na = length(state(env)), length(action_space(env))
    agent = Agent(
        policy = QBasedPolicy(
            learner = BasicDQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 64, relu; init = glorot_uniform(rng)),
                        Dense(64, 64, relu; init = glorot_uniform(rng)),
                        Dense(64, na; init = glorot_uniform(rng)),
                    ) |> gpu,
                    optimizer = ADAM(),
                ),
                batch_size = 32,
                min_replay_history = 100,
                loss_func = huber_loss,
                rng = rng,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ??_stable = 0.01,
                decay_steps = 500,
                rng = rng,
            ),
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 50_000,
            state = Vector{Float64} => (ns,),
        ),
    )

    stop_condition = StopAfterStep(800_000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()

    Experiment(agent, env, stop_condition, hook, "")
end
using Plots
ex = E`JuliaRL_BasicDQN_SpacecraftEnv`
run(ex)
plot(ex.hook.rewards)
