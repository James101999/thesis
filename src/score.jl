function score(graph)
     # d is the damping factor
     d = 0.85
     # score threshold
     ϵ = 0.0001
     # number of iterations
     iteration = 30
     # Intialize scores of each vertices to 1
     score = ones(length(graph.terms))
     neighborhood = [neighbors(graph, v) for v ∈ vertices(graph)]
     sizeofneighbors = map(length, neighborhood)
     previousscore = deepcopy(score)
     for iter = 1:iteration
         for V_i in neighborhood 
             ∑ = 0
             for V_j in i
                 ∑ = sum(1/neighbors(graph, sizeofneighbors[j])) * score[j]
         score[i] = (1 - d) + d * ∑ 
         # if at least one score in the graph falls below the threshold
end





