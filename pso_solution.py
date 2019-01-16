from pso import PSO

dim = 14
size = 8
iter_num = 20

pso = PSO(dim, size, iter_num)
fitness_list, best_position = pso.iterate()

print("best position:", best_position)
print("best fitness:", fitness_list[-1])
