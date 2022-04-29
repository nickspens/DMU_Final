using LinearAlgebra
using Test
using ReinforcementLearningBase.RLBase
using Random
using Random: AbstractRNG
export SpacecraftEnv


struct SpacecraftEnvParams{T}
    min_radius::T
    max_radius::T
    # max_speed::T
    goal_distance::T
    # goal_velocity::T # Add the velocity later
    Isp::T
    g0::T
    Thrust::T
    mu::T
    max_steps::Int
    timestep::Int
end

# Print object and parameters
Base.show(io::IO, params::SpacecraftEnvParams) = print(
    io,
    join(["$p=$(getfield(params, p))" for p in fieldnames(SpacecraftEnvParams)], ","),
)


function SpacecraftEnvParams(;
    T = Float64,
    min_radius = 6571, #km (200 km from Earth surface)
    max_radius = 7571, #km (1000 km above Earth surface)
    goal_distance = 0.05, #km (goal distance between sc0 and target spacecrafts)
    # goal_velocity = 7.3043, #km/s
    Isp = 300, #hydrogen engines
    g0 = 0.00981, #km/s
    Thrust = 20, #N
    mu = 398600.4, #km^3/s^2
    max_steps = 200,
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



# How could we define this mutable structure properly? 
mutable struct SpacecraftEnv{A,T,ACT,R<:AbstractRNG} <: AbstractEnv
    params::SpacecraftEnvParams{T}
    action_space::A
    sc_state::Vector{T}
    action::ACT
    reward::Float64
    done::Bool
    t::Int
    rng::R
end

"""
    MountainCarEnv(;kwargs...)
# Keyword arguments
- `T = Float64`
- `rng = Random.GLOBAL_RNG`
- `T = Float64,
- `min_radius = 6571, 
- `max_radius = 7571, 
- `goal_distance = 0.05, 
- `Isp = 300, 
- `g0 = 0.00981, 
- `Thrust = 20,
- `mu = 398600.4, 
- `max_steps = 200,
- `timestep` = 1,
"""
function SpacecraftEnv(;
    T = Float64,
    rng = Random.GLOBAL_RNG,
    kwargs...,
)
    params = SpacecraftEnvParams(; goal_distance = 0.05, Isp = 300, T = T, kwargs...)
    action_space = ([-1, 0, 1])
    reward = SpacecraftEnvReward(env, fuelReward, fstateRewards, distance) ## ASK PROF/TA
    env = SpacecraftEnv(
        params,
        action_space,
        zeros(4,6),
        #zeros(SC,5), ## DELETE
        #zeros(T,6), #radius, theta, vr, v_theta, mass, m_dot (kg) ## DELETE
        rand(action_space),
        reward,
        false,
        0,
        rng,
    )
    reset!(env)
    env
end

Random.seed!(env::SpacecraftEnv, seed) = Random.seed!(env.rng, seed)
RLBase.action_space(env::SpacecraftEnv) = env.action_space
# RLBase.reward(env::SpacecraftEnv{A,T}) where {A,T} =  env.done ? zero(T) : -one(T) 
RLBase.reward(env::SpacecraftEnv{A,T}) where {A,T} = env.reward
RLBase.is_terminated(env::SpacecraftEnv) = env.done
RLBase.state(env::SpacecraftEnv) = env.state

function RLBase.reset!(env::SpacecraftEnv{A,T}) where {A,T}

    # Servicing SC
    env.sc_state[0, 0] = 6571; #km
    env.sc_state[0, 1] = 0.0; #rad
    env.sc_state[0, 2] = 0.0; #km/s
    env.sc_state[0, 3] = 7.7885; #km/s
    env.sc_state[0, 4] = 1000.0; #kg
    env.sc_state[0, 5] = 0; #kg/s


    # SC 1 - to avoid
    env.sc_state[1, 0] = 6771; #km
    env.sc_state[1, 1] = 0.7853; #rad
    env.sc_state[1, 2] = 0.0; #km/s
    env.sc_state[1, 3] = 7.6726; #km/s
    env.sc_state[1, 4] = 1000.0; #kg
    env.sc_state[1, 5] = 0; #kg/s
    
    # SC 2 - to avoid
    env.sc_state[2, 0] = 6971; #km
    env.sc_state[2, 1] = 1.5708; #rad
    env.sc_state[2, 2] = 0.0; #km/s
    env.sc_state[2, 3] = 7.5617; #km/s
    env.sc_state[2, 4] = 1000.0; #kg
    env.sc_state[2, 5] = 0; #kg/s

    # SC 3 - to be refuelled
    env.sc_state[3, 0] = 7171; #km
    env.sc_state[3, 1] = 2.5691; #rad
    env.sc_state[3, 2] = 0.0; #km/s
    env.sc_state[3, 3] = 7.4555; #km/s
    env.sc_state[3, 4] = 1000.0; #kg
    env.sc_state[3, 5] = 0; #kg/s

    # Space Station - to be refuelled
    env.sc_state[4, 0] = 7471; #km
    env.sc_state[4, 1] = 4.7125; #rad
    env.sc_state[4, 2] = 0.0; #km/s
    env.sc_state[4, 3] = 7.3043; #km/s
    env.sc_state[4, 4] = 1000.0; #kg
    env.sc_state[4, 5] = 0; #kg/s


    env.done = false
    env.t = 0
    nothing
end

function _step!(env::SpacecraftEnv)

    distance, fuelcost = state_update(env);
    
    env.done = env.t >= env.params.max_steps||
        for i in 1:4
            if dist[i] <= env.goal_distance;
            # && norm([sc[0].state[2],sc[0].state[3]]) == env.goal_velocity
                if i==1
                    fstateReward = -100;
                elseif i==2
                    fstateReward = -60;
                elseif i==3
                    fstateReward = 50;
                    env.done = false; #reaching this sc is not a final state
                elseif i==4
                    fstateReward = 100;
                end
            end
        end
    
    env.reward = SpacecraftEnvReward(env, fuelcost, fstateReward, distance) # Ask Prof/TA can we call this in this way?

    nothing
end

function state_update(env::SpacecraftEnv)
    μ = env.mu;
    sc = env.sc_state;
    throttle = env.action;
    Thr = env.Thrust*throttle;
    env.t += env.timestep;

    for i in 0:4
        r,θ,vr,vθ,m,m_dot = sc_state[i, :];
        r_dot = vr ;
        θ_dot = vθ/r; 
        vr_dot = vθ^2/r - μ/(r^2);
        vθ_dot = -vr*vθ/r;

        r_new = r+r_dot*timestep;
        θ_new = θ+θ_dot*timestep;
        vr_new = vr+vr_dot*timestep;
        vθ_new = vθ+vθ_dot*timestep;
        m = 1000;
        m_dot = 0;
        
        if i==0
            m_dot = -Thr/(env.Isp*g0);
            m_new = m-m_dot*timestep;
            vθ_dot = -vr*vθ/r + Thr/(m-m_dot*timestep);
            fuelcost = m_new - m;
        end

        sc_state[i, :] = [r_new,θ_new,vr_new,vθ_new,m_new,m_dot];

        distance = distance(sc_state)
        
        return(distance, fuelcost)

    end

    function distance(sc_state)

        r0 = sc_state[0, 1]
        θ0 = sc_state[0, 2]

        for i in 1:4
            r = sc_state[i, 1]
            θ = sc_state[i, 2]

            distance = sqrt((r*cos(θ) - r0*cos(θ0))^2 + (r*sin(θ) - r0*sin(θ0))^2)

            push!(distance_s0, distance)
        end
        return(distance_s0)
    end

end

function SpacecraftEnvReward(env::SpacecraftEnv,fuelcost, fstateReward, distance)

    d = distance;
    weight = [30, 20, 20, 40]; #TBD
    contReward = dot([-1/d[0], - 1/d[1], 1/d[2], 1/d[3]],weight); #negative rewards for obj 1 and 2 (in sc vector), positive reward for obj 3 and 4 (in sc vector)
    
    tot_reward = contReward + fuelcost + fstateReward;

    return (tot_reward)
end
