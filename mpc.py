class MPC():
    def __init__(self, path, horizon, dt):
        self.path = path
        self.horizon = horizon
        self.dt = dt
        self.bnds = jnp.concatenate((jnp.array([(-30, 30)] * 4 * horizon), jnp.array([(-100, 100)] * 6 * horizon)))
        self.prev_optimization = jnp.array([0] * self.horizon * 10)

        self.obj_jit = 0
        self.obj_grad = 0
        self.obj_hess = 0
        self.objJitSetup()

    def predictMPC(self, prev_state, u):
        dt = self.dt
        local_vel = jnp.array([[prev_state[3], prev_state[4], prev_state[5]]]).T
        global_vel = rotationMatrixJ(prev_state[2]) @ local_vel
        
        xPos = prev_state[0] + dt * global_vel[0][0]
        yPos = prev_state[1] + dt * global_vel[1][0]
        theta = prev_state[2] + dt * global_vel[2][0]
        M = (1/r_dis) * jnp.array([[1, -1, -(L_dis + l_dis)],
                          [1, 1, (L_dis + l_dis)], 
                          [1, 1, -(L_dis + l_dis)],
                          [1, -1, (L_dis + l_dis)]])
        M_inv = jnp.linalg.pinv(M)
        M_inv_u = M_inv @ u
        c1 = M_inv_u[0]
        c2 = M_inv_u[1]
        c3 = M_inv_u[2]
        
        xVel = c1 #c1 * jnp.cos(prev_state[2]) - c2 * jnp.sin(prev_state[2])
        yVel = c2# c1 * jnp.sin(prev_state[2]) + c2 * jnp.cos(prev_state[2])
        thetaVel = c3        
        return jnp.array([xPos, yPos, theta, xVel, yVel, thetaVel])

    def objective(self, x, state):
        horizon = self.horizon
        u_list = x[:4 * horizon]
        target_states = self.getTargetTrajectory(state, 60)
        print("target states: " + str(target_states))
        state_cost = 0
        model_constraint = 0
        
        for i in range(horizon):
            print("i: " + str(i))

            if (i >= 1):
                prev_state = x[horizon * 4 + 6 * (i-1) : horizon * 4 + 6 * i]
                current_state = x[horizon * 4 + 6 * i : horizon * 4 + 6 * (i+1)]
                prev_u = x[4 * (i-1) : 4 * i]
                predicted_current = self.predictMPC(prev_state, prev_u)
                
                model_constraint += jnp.sum((predicted_current - current_state) ** 2)
                
            position_state = x[horizon * 4 + i * 6 : horizon * 4 + i * 6 + 3]
            target_position = target_states[i]
            target_position = jnp.array([target_position[0], target_position[1], self.path[-1][2]])
            print("target pos: " + str(target_position))
            print("pos state: " + str(position_state))
            position_cost = jnp.sum((target_position[:2] - position_state[:2]) ** 2)
            angle_cost = 1 * (target_position[2] - position_state[2]) ** 2
            state_cost += (position_cost + angle_cost)

        control_cost = jnp.sum(u_list ** 2)
        
        total_cost = 1 * state_cost + 9999 * model_constraint + 0 * control_cost
        print("cost: " + str(total_cost))
        return total_cost
    
    def getTargetTrajectory(self, prev_state, vel):
        #return jnp.array([[50, 50, 0], [50, 50, 0], [50, 50, 0], [50, 50, 0]])
        
        #really sketchy append
        #'''
        [projection, pathIndex] = self.getClosestPoint(prev_state)
        
        altered_path = self.path
        altered_path = altered_path[pathIndex:]
        arr = [projection]
        for point in altered_path:
            print(point)
            arr.append(point)
        altered_path = arr
        target = self.moveAlongPath(altered_path, vel)
        return target
        #'''
        
    def moveAlongPath(self, path, vel):
        segment_lengths = []
        target_trajectory = []
        atEnd = False
        for i in range(len(path) - 1):
            segment_lengths.append(distance(path[i], path[i+1]))
            
        for i in range(self.horizon):
            segment_distance = vel * (i+1) * self.dt
            j = 0
            while (segment_distance > 0):
                if (j >= len(segment_lengths)):
                    target_trajectory.append(path[-1])
                    #print("at end")
                    atEnd = True
                    break
                segment_distance -= segment_lengths[j]
                j += 1
            if (not(atEnd)):
                segment_distance += segment_lengths[j-1]
                current_segment = Segment(path[j-1], path[j])

                target_trajectory.append(current_segment.moveAlongSegment(segment_distance))
            
        return target_trajectory

    def getClosestPoint(self, prev_state):
        path = self.path
        minDistance = 9999999.
        dis = 0.
        closestPoint = [0, 0]
        pathIndex = 1
        for i in range(len(path) - 1):
            currentSegment = Segment(path[i], path[i+1])
            projection = currentSegment.projectToSegment(prev_state)
            dis = distance(prev_state, projection)
            if (dis < minDistance):
                minDistance = dis
                closestPoint = projection
                pathIndex = i+1
        return [closestPoint, pathIndex]
        
    def objJitSetup(self):
        self.obj_jit = jit(self.objective)
        self.obj_grad = jit(grad(self.obj_jit))
        self.obj_hess = jit(jacrev(jacfwd(self.obj_jit)))

    def step(self, prev_state):
        horizon = self.horizon
        dtMPC = self.dt

        eq_constraints = lambda x, prev_state=prev_state : np.sum((x[4 * horizon : 4 * horizon + 6] - prev_state) ** 2)
        cons_jit = jit(eq_constraints)
        cons_jac = jit(jacfwd(cons_jit))
        cons_hess = jacrev(jacfwd(cons_jit))
        cons_hessvp = jit(lambda x, v, cons_hess=cons_hess: cons_hess(x) * v[0])

        cons = [{'type': 'eq', 'fun': cons_jit, 'jac': cons_jac, 'hess': cons_hessvp}]

        
        # loops = 2
        # test_cat = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        # print("controls test:" + str(test_cat[4:4*loops]))
        # print("states test:" + str(test_cat[4*loops + 6:]))
        # print("optimize test:" + str(np.concatenate((test_cat[4:4*loops], test_cat[4*loops + 6:]), axis=0)))
        
        prev_opt = self.prev_optimization
        cdr_controls = prev_opt[4 : 4*horizon]
        cdr_states = prev_opt[4*horizon+6:]
        x0 = jnp.concatenate((jnp.concatenate((cdr_controls, prev_opt[4*horizon - 4 : 4*horizon])),
                              jnp.concatenate((cdr_states, prev_opt[10*horizon - 6:]))))
        
        res = minimize_ipopt(self.obj_jit, jac=self.obj_grad, hess=self.obj_hess, x0=x0, bounds=self.bnds,
                constraints=cons, options={"max_iter": 300, "acceptable_iter": 1, "constr_viol_tol": 1e-1,
                                           "acceptable_tol": 1e-1, "tol": 1e-2})
        print("")
        print("target trajectory: " + str(self.getTargetTrajectory(prev_state, 120)))
        #print("minimized value: " + str(res.fun))
        optimization = res.x
        print("optimization: " + str(optimization))
        self.prev_optimization = optimization
        #MPC_trajectory = optimization[4 * horizon:]
        #print("mpc trajectory: " + str(MPC_trajectory))
        return np.array([optimization[4:8]]).T
test_arr = [[0, 0], [1, 1], [2, 2]]
test_arr.insert(0, [3, 3])
#print(test_arr)
test_path = [[0, 0], [1, 1], [1, 2]]
testMPC = MPC(test_path, 4, 1)
#print(testMPC.getTargetTrajectory([0.5, 0.5] , 1))

    
targets = []

#print("bnds: " + str(bnds))


#initial_state = np.array([13.75/2, 130, np.pi/2, 0, 0, 0]) #cycle path
mpc_horizon = 4

#paths to test on
CYCLE_HUB = np.array([[13.75/2, 130, np.pi/2], [13.75/2, 80, np.pi/2], [35, 65, 3*np.pi/4]]) #cycle path
CYCLE_WAREHOUSE = np.array([[35, 65, 3*np.pi/4], [13.75/2, 80, np.pi/2], [13.75/2, 130, np.pi/2]]) #cycle path

#path = np.array([[0, 0, 0], [50, 50, 0]]) #diagonal path
#path = np.array([[0, 0, 0], [50, 0, 0], [50, 50, np.pi/2], [0, 50, np.pi]])
test_path = np.array([[0, 0, 0], [50, 0, 99999], [50, 50, 99999], [0, 50, np.pi]])

#MPC
#mpc = MPC(current_path, mpc_horizon, mpc_dTime)
#mpc.objective(jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

