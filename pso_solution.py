from pso import PSO

dim = 14
size = 8
iter_num = 20

pso = PSO(dim, size, iter_num)
fitness_list, best_position = pso.iterate()

with open("pso_result.txt", "w") as f:
    f.write("best position: " + str(best_position) + "\n")
    f.write("best fitness: " + str(fitness_list[-1]) + "\n")
    f.write("history fitness:\n")
    for i in range(len(fitness_list)):
        f.write(str(fitness_list(i)) + "\n")

print("best position:", best_position)
print("best fitness:", fitness_list[-1])
