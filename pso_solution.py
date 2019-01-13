

from pso import PSO

dim = 16
size = 20
iter_num = 100

pso = PSO(dim, size, iter_num)
fitness_list, best_position = pso.iterate()

print("best position:", best_position)
print("best fitness:", fitness_list[-1])
