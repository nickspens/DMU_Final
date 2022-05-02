

function DQN(env)
    
    # This network should work for the Q function - an input is a state; the output is a vector containing the Q-values for each action 
    Q = Chain(Dense(12, 128, relu),Dense(128,128,relu), Dense(128, length(env.action_space)))
    Q_best = deepcopy(Q)
    Qp = deepcopy(Q)
    # Evaluate the new Q and substitute the Q_best if it performs better
    # rew_Qbest = HW5.evaluate(s->actions(env)[argmax(Q_best(s[1:2]))], n_episodes=100)
    best_action = Float64[]  
    mean_reward = NamedTuple{(:score,), Tuple{Float64}}[]
    iter = 1
    
    for i=1:700

        #Exploration policy
        function policy(s,ϵ,Q)
            # if rand() < ϵ
            #     if s[2]<0
            #         return actions(env)[1]
            #     elseif s[2]>=0
            #         return actions(env)[2]
            #     end      
            # else
            #     return argmax(a->Q(s), actions(env))
            # end
            return env.action_space[1]
        end
        
        #Create the experience buffer: update it every 10 epochs
        ϵ = LinRange(0.5, 0.05, 701) #mod
        if i ==1 || i%10 ==0
            global buffer =Tuple{Vector{Float64}, Int64, Float64, Vector{Float64}, Bool}[]
            for j = 1:50
                reset!(env)
                done = env.done
                while done != true
                    # Tuple of experience like this
                    s = env.state
                    a = policy(s,ϵ[iter],Q)
                    a_ind = findall(x->x==a, env.action_space)[1] # action index 
                    r = _step!(env)
                    sp = env.state
                    done = env.done
                    experience_tuple = (s, a_ind, r, sp, done)
                    push!(buffer,experience_tuple)
                    
                end
            end
        else
            for j = 1:50
            reset!(env)
            done = env.done
            while done != true
                # Tuple of experience like this
                s = env.state
                a = policy(s,ϵ[iter],Q)
                a_ind = findall(x->x==a, env.action_space)[1] # action index 
                r =  _step!(env)
                sp = env.state
                done = env.done
                experience_tuple = (s, a_ind, r, sp, done)
                push!(buffer,experience_tuple)
            end
            end
            iter+=1
        end

        γ=0.99
        
        if i ==1 || i%5 ==0
            Qp = deepcopy(Q)
        end
        
        # Loss function
        function loss(s, a_ind, r, sp, done)
            if done
                return (r-Q(s)[a_ind])^2
            else
                return (r+0.99*maximum(Qp(sp))-Q(s)[a_ind])^2 
            end
        end

        # select some data from the buffer
        data = rand(buffer, 10_000)
        
        
        # Train the Neural Network
        Flux.Optimise.train!(loss, params(Q), data, ADAM(0.0005))#mod
        
        #Evaluate Q after each training and save the mean reward
        # Q_reward=HW5.evaluate(s->actions(env)[argmax(Q(s[1:2]))], n_episodes=100)
        Q_reward = run(s->env.action_space[argmax(Q(s[1:12]))],env,StopAfterStep(1000),TotalRewardPerEpisode())
        push!(mean_reward, Q_reward)
         
        # If Q performs better than Q_best, substitute
        if Q_reward > rew_Qbest
            Q_best = Q
            rew_Qbest = Q_reward
            print("Best Q found:\n")
            print(rew_Qbest)
            if rew_Qbest[1] >40
                # HW5.evaluate(s->actions(env)[argmax(Q_best(s[1:2]))],"giuliana.miceli@colorado.edu"; n_episodes=10_000,fname="first_res.json")
                best_action = [env.action_space[argmax(Q_best(s[1:2]))]]
            end
        end
        
    end
    return (mean_reward, Q_best, best_action)
end

# tot_reward, Q_best, best_act = DQN(env)
