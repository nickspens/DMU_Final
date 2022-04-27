using LinearAlgebra
export SpacecraftEnv


struct SpacecraftEnvParams{T}
    min_radius::T
    max_radius::T
    # max_speed::T
    goal_distance::T
    # goal_velocity::T # Add the velocity later
    Isp::T
    g0::T
    mu::T
    max_steps::Int
    timestep::Int
end

# Ask professor: is it for printing the parameters?
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
    mu = 398600.4, #km^3/s^2
    max_steps = 200,
    timestep = 1,
)
    MountainCarEnvParams{T}(
        min_radius,
        max_radius,
        goal_distance,
        # goal_velocity,
        Isp,
        g0,
        mu,
        max_steps,
        timestep,
    )
end

function SpacecraftEnvReward(env::SpacecraftEnv)
    fuelReward = sc[0].fuelReward;
    d = env.sc.distance;
    weight = [30, 20, 20, 40]; #TBD
    contReward = dot([-1/d[0], - 1/d[1], 1/d[2], 1/d[3]],weight);
    
    env.reward = contReward + fuelReward + fstateReward;

end


mutable struct SpacecraftEnv{A,T,ACT,R<:AbstractRNG} <: AbstractEnv
    params::SpacecraftEnvParams{T}
    action_space::A
    state::Vector{T}
    action::ACT
    rewards::Float64
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
    reward = SpacecraftEnvReward(env)
    env = SpacecraftEnv(
        params,
        action_space,
        zeros(T,6), #radius, theta, vr, v_theta, mass, m_dot (kg)
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
RLBase.reward(env::SpacecraftEnv{A,T}) where {A,T} =  env.done ? zero(T) : -one(T) ## ASK PROFESSOR 
# Dict(env.sc.distance[1]<= env.goal_distance => -100.0,
# env.sc.distance[2]=> -60.0,
# env.sc.distance[3]=> 50.0,
# env.sc.distance[4]=> 100.0,
# env.action[0]=> -env.sc.state[5],
# env.action[2]=> -env.sc.state[5]),
RLBase.is_terminated(env::SpacecraftEnv) = env.done
RLBase.state(env::SpacecraftEnv) = env.state

function RLBase.reset!(env::SpacecraftEnv{A,T}) where {A,T}

    # Servicing SC
    env.sc[0].state[0] = 6571; #km
    env.sc[0].state[1] = 0.0; #rad
    env.sc[0].state[2] = 0.0; #km/s
    env.sc[0].state[3] = 7.7885; #km/s
    env.sc[0].state[4] = 1000.0; #kg
    env.sc[0].state[5] = 0; #kg/s

    # SC 1 - to avoid
    env.sc[1].state[0] = 6771; #km
    env.sc[1].state[1] = 0.7853; #rad
    env.sc[1].state[2] = 0.0; #km/s
    env.sc[1].state[3] = 7.6726; #km/s
    env.sc[1].state[4] = 1000.0; #kg
    env.sc[1].state[5] = 0; #kg/s

    # SC 2 - to avoid
    env.sc[2].state[0] = 6971; #km
    env.sc[2].state[1] = 1.5708; #rad
    env.sc[2].state[2] = 0.0; #km/s
    env.sc[2].state[3] = 7.5617; #km/s
    env.sc[2].state[4] = 1000.0; #kg
    env.sc[2].state[5] = 0; #kg/s

    # SC 3 - to be refuelled
    env.sc[3].state[0] = 7171; #km
    env.sc[3].state[1] = 2.5691; #rad
    env.sc[3].state[2] = 0.0; #km/s
    env.sc[3].state[3] = 7.4555; #km/s
    env.sc[3].state[4] = 1000.0; #kg
    env.sc[3].state[5] = 0; #kg/s

    # Space Station - to be refuelled
    env.sc[4].state[0] = 7471; #km
    env.sc[4].state[1] = 4.7125; #rad
    env.sc[4].state[2] = 0.0; #km/s
    env.sc[4].state[3] = 7.3043; #km/s
    env.sc[4].state[4] = 1000.0; #kg
    env.sc[4].state[5] = 0; #kg/s


    env.done = false
    env.t = 0
    nothing
end

function _step!(env::SpacecraftEnv)

    state_update(env.sc, env.action, env.timestep);
    
    env.done = env.t >= env.params.max_steps||
        for i in 1:4
            if sc[0].distance[i] <= env.goal_distance;
            # && norm([sc[0].state[2],sc[0].state[3]]) == env.goal_velocity
                if i==1
                    fstateReward = -100;
                elseif i==2
                    fstateReward = -60;
                elseif i==3
                    fstateReward = 50;
                elseif i==4
                    fstateReward = 100;
                end
            end
        end
    return(fstateReward)
    
end

function state_update(env.sc,throttle = env.action, env.timestep)  # Can we take into account the actions in this way? Can we call the spacecraft in this way?
    μ = env.mu;
    Thr = env.T*throttle;


    env.t += timestep;

    for i in sc
        r,θ,vr,vθ,m,m_dot = sc[i].state;
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
            fuelReward = m_new - m;
        end

        sc[i].state = [r_new,θ_new,vr_new,vθ_new,m_new,m_dot];
        sc[0].fuelReward = fuelReward;
        distance(sc)
        env.reward = SpacecraftEnvReward(env)
        
    end

    function distance(env.sc)

        r0= sc[0].state[1]
        θ0 = sc[0].state[2]

        for i in 1:4
            r = sc[i].state[1]
            θ = sc[i].state[2]

            distance = sqrt((r*cos(θ) - r0*cos(θ0))^2 + (r*sin(θ) - r0*sin(θ0))^2)

            sc[0].distance[i] = distance
        end
    end




end
