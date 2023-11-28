def distance_levenstein(s1,s2):
    m = len(s1)
    n = len(s2)
    
    mat_dist = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        mat_dist[i][0] = i
    
    for j in range(n + 1):
        mat_dist[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cout = 0 if s1[i - 1] == s2[j - 1] else 1
            mat_dist[i][j] = min(
                mat_dist[i - 1][j - 1] + cout,
                min(mat_dist[i - 1][j] + 1, mat_dist[i][j - 1] + 1)
            )
    
    answer = mat_dist[m][n]
    
    return answer