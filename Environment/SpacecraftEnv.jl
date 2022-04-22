export SpacecraftEnv, ContinuousSpacecraftEnv

struct SpacecraftEnvParams{T}
    min_radius::T
    max_radius::T
    # max_speed::T
    goal_distance::T
    goal_velocity::T
    Isp::T
    g0::T
    mu::T
    max_steps::Int
end

Base.show(io::IO, params::SpacecraftEnvParams) = print(
    io,
    join(["$p=$(getfield(params, p))" for p in fieldnames(SpacecraftEnvParams)], ","),
)

function SpacecraftEnvParams(;
    T = Float64,
    min_radius = 6571, #km
    max_radius = 7571, #km
    goal_distance = 0.05, #km
    goal_velocity = 7.5617, #km/s
    Isp = 300, #hydrogen engines
    g0 = 0.00981, #km/s
    mu = 398600.4, #km^3/s^2
    max_steps = 200,
)
    MountainCarEnvParams{T}(
        min_radius,
        max_radius,
        goal_distance,
        goal_velocity,
        Isp,
        g0,
        mu,
        max_steps,

    )
end

mutable struct SpacecraftEnv{A,T,ACT,R<:AbstractRNG} <: AbstractEnv
    params::SpacecraftEnvParams{T}
    action_space::A
    # observation_space::Space{Vector{ClosedInterval{T}}}
    state::Vector{T}
    action::ACT
    done::Bool
    t::Int
    rng::R
end

"""
    MountainCarEnv(;kwargs...)
# Keyword arguments
- `T = Float64`
- `continuous = false`
- `rng = Random.GLOBAL_RNG`
- `T = Float64,
- `min_radius = 6571, 
- `max_radius = 7571, 
- `goal_distance = 0.05, 
- `goal_velocity = 7.5617, 
- `Isp = 300, 
- `g0 = 0.00981, 
- `mu = 398600.4, 
- `max_steps = 200,
"""
function SpacecraftEnv(;
    T = Float64,
    # continuous = false,
    rng = Random.GLOBAL_RNG,
    kwargs...,
)
    # if continuous
    params = SpacecraftEnvParams(; goal_distance = 0.05, Isp = 300, T = T, kwargs...)
    # else
    #     params = SpacecraftEnvParams(; T = T, kwargs...)
    # end
    action_space = (-1, 0, 1)
    env = SpacecraftEnv(
        params,
        action_space,
        # Space([params.min_pos..params.max_pos, -params.max_speed..params.max_speed]),
        zeros(T,3), #radius, theta, mass (kg)
        rand(action_space),
        false,
        0,
        rng,
    )
    reset!(env)
    env
end

ContinuousSpacecraftEnv(; kwargs...) = SpacecraftEnv(; continuous = true, kwargs...)

Random.seed!(env::SpacecraftEnv, seed) = Random.seed!(env.rng, seed)

RLBase.action_space(env::SpacecraftCarEnv) = env.action_space
RLBase.reward(env::SpacecraftEnv{A,T}) where {A,T} = Dict(env.sc.distance[1]=> -100.0,
env.sc.distance[2]=> -100.0,
env.sc.distance[3]=> 50.0,
env.sc.distance[4]=> 100.0,
env.action[0]=> -env.sc.dM,
env.action[2]=> -env.sc.dM)
RLBase.is_terminated(env::SpacecraftEnv) = env.done
RLBase.state(env::SpacecraftEnv) = env.state

function RLBase.reset!(env::SpacecraftEnv{A,T}) where {A,T}
    env.state[1] = 6571 #km
    env.state[2] = 0.0 #rad
    env.state[3] = 7.7885 #km/s
    env.state[4] = 0.0 #km/s
    env.state[5] = 1000.0 #kg
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
    _step!(env, a - 2)
end

function _step!(env::SpacecraftEnv, Thrust)
    timestep = 1

    state_update(env.sc[0],Thrust)

    env.sc[0].state
    env.sc[0].distance



    #env.done =
    #    r >= env.params.goal_pos && v >= env.params.goal_velocity ||
    #    env.t >= env.params.max_steps

    nothing
end

function state_update(spacecraft,Thrust)
    r,θ,vr,vθ,m,m_dot = spacecraft.state
    μ = env.mu
    T = env.T*Thrust

    m_dot = -T/(I_sp*g0)

    r_dot = vr 
    θ_dot = vθ/r 
    vr_dot = vθ^2/r - μ/(r^2)
    vθ_dot = -vr*vθ/r + T/(m-m_dot*timestep)


    env.t += timestep
    r_new = r+r_dot*timestep
    θ_new = θ+θ_dot*timestep
    vr_new = vr+vr_dot*timestep
    vθ_new = vθ+vθ_dot*timestep
    m_new = m-m_dot*timestep

    spacecraft.state = [r_new,θ_new,vr_new,vθ_new,m_new,m_dot]
end
