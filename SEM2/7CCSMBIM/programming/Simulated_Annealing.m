ObjectiveFunction = @simple_objective;
x0 = [0.5 0.5];
rng default
[x,fval,exitFlag,output] = simulannealbnd(ObjectiveFunction,x0);