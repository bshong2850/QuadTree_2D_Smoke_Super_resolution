def solution_1(N, mine):

    def check_mine(total_map, m_s):
        total_map[m_s[0] - 1][m_s[1] - 1] = -1
        if m_s[1] - 2 >= 0 and total_map[m_s[0] - 1][m_s[1] - 2] != -1:
            total_map[m_s[0] - 1][m_s[1] - 2] += 1
        if m_s[0] - 2 >= 0 and total_map[m_s[0] - 2][m_s[1] - 1] != -1:
            total_map[m_s[0] - 2][m_s[1] - 1] += 1
        if m_s[0] - 2 >= 0 and m_s[1] - 2 >= 0 and total_map[m_s[0] - 2][m_s[1] - 2] != -1:
            total_map[m_s[0] - 2][m_s[1] - 2] += 1

        if m_s[0] <= N - 1 and total_map[m_s[0]][m_s[1] - 1] != -1:
            total_map[m_s[0]][m_s[1] - 1] += 1
        if m_s[1] <= N - 1 and total_map[m_s[0] - 1][m_s[1]] != -1:
            total_map[m_s[0] - 1][m_s[1]] += 1
        if m_s[1] <= N - 1 and m_s[0] <= N - 1 and total_map[m_s[0]][m_s[1]] != -1:
            total_map[m_s[0]][m_s[1]] += 1

        if m_s[1] <= N - 1 and m_s[0] - 2 >= 0 and total_map[m_s[0] - 2][m_s[1]] != -1:
            total_map[m_s[0] - 2][m_s[1]] += 1
        if m_s[0] <= N - 1 and m_s[1] - 2 >= 0 and total_map[m_s[0]][m_s[1] - 2] != -1:
            total_map[m_s[0]][m_s[1] - 2] += 1

    total_map = [[0 for _ in range(N)] for _ in range(N)]

    while mine:
        mine_single = mine.pop()
        check_mine(total_map, mine_single)

    return total_map


#solution(9, [ [1, 1], [1, 7], [2, 7], [3, 6], [4, 1], [4, 4], [4, 8], [8, 4], [8, 5], [9, 6] ])

import copy

def solution_2(A, S):
    A_max = max(A)
    if A_max > S:
        return 1

    else:
        for i in range(2, len(A) + 1):
            if i != len(A):
                A_copy = copy.deepcopy(A)
                list = []
                for _ in range(i):
                    list.append(A_copy.pop(0))
                while A_copy:
                    print(i, list, sum(list))
                    if sum(list) >= S:
                        return print(i)
                    else:
                        list.pop(0)
                        list.append(A_copy.pop(0))
            else:
                if sum(A) >= S:
                    return print(len(A))


        return print(-1)

#solution([1, 10, 2, 9, 3, 8, 4, 7, 5, 6], 55)

from itertools import product
def solution_3(board):
    N = len(board)

    x_list = [[i for i in range(N)]] * 2

    total_list = list(product(*x_list))
    rook_total_list = []
    x_list_copy = copy.deepcopy(total_list)
    for _ in range(N):
        rook_list = []
        x_index = []
        y_index = []
        x_list_single = x_list_copy.pop(0)
        rook_list.append(x_list_single)
        x_index.append(x_list_single[0])
        y_index.append(x_list_single[1])
        x_list_copy_copy = copy.deepcopy(x_list_copy)
        while x_list_copy_copy:
            x_list_single_ = x_list_copy_copy.pop(0)
            if x_list_single_[0] not in x_index and x_list_single_[1] not in y_index:
                rook_list.append(x_list_single_)
                x_index.append(x_list_single_[0])
                y_index.append(x_list_single_[1])

        rook_total_list.append(rook_list)



    print(rook_total_list)

    sum_rook = []
    while rook_total_list:
        rook_list_single = rook_total_list.pop()
        sum = 0
        while rook_list_single:
            rook_list_single_s = rook_list_single.pop()
            sum += board[rook_list_single_s[0]][rook_list_single_s[1]]

        sum_rook.append(sum)


    print(sum_rook)


    answer = 0
    return answer

#solution_3([[3,6,8],[1,4,7],[2,1,4]])



from itertools import permutations
def solution_4(board):
    N = len(board)
    x_list = [i for i in range(N)]

    xx = list(permutations(x_list, N))
    rook_total_list = []
    while xx:
        rook_list_s = []
        xx_ = xx.pop(0)
        for i in range(N):
            rook_list_s.append([i, xx_[i]])
        rook_total_list.append(rook_list_s)

    sum_rook = []
    while rook_total_list:
        rook_list_single = rook_total_list.pop()
        sum = 0
        while rook_list_single:
            rook_list_single_s = rook_list_single.pop()
            sum += board[rook_list_single_s[0]][rook_list_single_s[1]]

        sum_rook.append(sum)

    print(max(sum_rook))

solution_4([[3, 6, 8], [1, 4, 7], [2, 1, 4]])
solution_4([[12,15],[19,21]])

solution_4([[1,2,3,4,5],[1,5,3,4,2],[1,2,5,4,3],[1,2,3,5,4],[5,2,3,4,1]])














































