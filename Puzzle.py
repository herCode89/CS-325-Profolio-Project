from typing import List, Any

Board = [["-", "-", "-", "-", "-"],
         ["-", "-", "#", "-", "-"],
         ["-", "-", "-", "-", "-"],
         ["#", "-", "#", "#", "-"],
         ["-", "#", "-", "-", "-"]]


def solve_puzzle(Board, Source, Destination):
    """
    In order to traversal this graph there will need to be the DFS approach.
    Call all functions to calculate the minimum length in order to filter the minimum routes possible
    to take. We want to visit all the cells while having a way to store the one directions in which were
    visited. We can then take the calculations of the DFS to determine the movements of Up, Down, Left, and Right
    """
    PathofBoard(Board, Source[0], Source[1], Destination, '')
    length = float(64)
    # calculate the minimum length
    for i in result:
        length = min(length, len(i))
    orig: list[Any] = []
    # length dir
    for i in result:
        if not len(i) != length:
            orig.append(i)

    if not len(orig):
        print(None)
        return

    print("Output:", len(orig) + 1)
    print("One possible direction to travel:", orig[0])


# To check the Board
def CheckBoard(Board, i, j):
    return \
        0 <= i < len(Board) and \
        0 <= j < len(Board[i]) and \
        not Board[i][j] != '-'


result: list[Any] = []  # Storing


def PathofBoard(Board, i, j, path, dir):
    if not CheckBoard(Board, i, j):
        return

    if not (i, j) != path:
        result.append(dir)
        return

    Board[i][j] = '#'  # Visited and Cited
    # Perform the direction in Left, Right, UP, Down
    if CheckBoard(Board, i, j - 1):
        PathofBoard(Board, i, j - 1, path, dir + 'L')

    if CheckBoard(Board, i, j + 1):
        PathofBoard(Board, i, j + 1, path, dir + 'R')

    if CheckBoard(Board, i - 1, j):
        PathofBoard(Board, i - 1, j, path, dir + 'U')

    if CheckBoard(Board, i + 1, j):
        PathofBoard(Board, i + 1, j, path, dir + 'D')
    Board[i][j] = '-'


if __name__ == '__main__':
    (a, b) = (1, 3)
    (x, y) = (3, 3)
    solve_puzzle(Board, (a - 1, b - 1), (x - 1, y - 1))

# (a1, b1) = (1, 1)
# (x1, y1) = (5, 5)
# solve_puzzle(Board, (a1 - 1, b1 - 1), (x1 - 1, y1 - 1))

# (a2, b2) = (1, 1)
# (x2, y2) = (5, 1)
# solve_puzzle(Board, (a2 - 1, b2 - 1), (x2 - 1, y2 - 1))
print("The time Complexity for this is O(3^(n^2))")

'''
Cited Sources:
Author: divyesh072019
URL: https://www.geeksforgeeks.org/boggle-find-possible-words-board-characters/
Code:
# Python3 program for Boggle game
# Let the given dictionary be following

dictionary = ["GEEKS", "FOR", "QUIZ", "GO"]
n = len(dictionary)
M = 3
N = 3

# A given function to check if a given string
# is present in dictionary. The implementation is
# naive for simplicity. As per the question
# dictionary is given to us.
def isWord(Str):

    # Linearly search all words
    for i in range(n):
        if (Str == dictionary[i]):
            return True
    return False

# A recursive function to print all words present on boggle
def findWordsUtil(boggle, visited, i, j, Str):
    # Mark current cell as visited and
    # append current character to str
    visited[i][j] = True
    Str = Str + boggle[i][j]

    # If str is present in dictionary,
    # then print it
    if (isWord(Str)):
        print(Str)

    # Traverse 8 adjacent cells of boggle[i,j]
    row = i - 1
    while row <= i + 1 and row < M:
        col = j - 1
        while col <= j + 1 and col < N:
            if (row >= 0 and col >= 0 and not visited[row][col]):
                findWordsUtil(boggle, visited, row, col, Str)
            col+=1
        row+=1

    # Erase current character from string and
    # mark visited of current cell as false
    Str = "" + Str[-1]
    visited[i][j] = False

# Prints all words present in dictionary.
def findWords(boggle):

    # Mark all characters as not visited
    visited = [[False for i in range(N)] for j in range(M)]

    # Initialize current string
    Str = ""

    # Consider every character and look for all words
    # starting with this character
    for i in range(M):
      for j in range(N):
        findWordsUtil(boggle, visited, i, j, Str)

# Driver Code
boggle = [["G", "I", "Z"], ["U", "E", "K"], ["Q", "S", "E"]]

print("Following words of", "dictionary are present")
findWords(boggle)
-----------------------------------------
Author: Kaustav kumar Chanda
URL: https://www.geeksforgeeks.org/minimum-steps-reach-target-knight/
Code: 
class cell:

    def __init__(self, x = 0, y = 0, dist = 0):
        self.x = x
        self.y = y
        self.dist = dist

# checks whether given position is
# inside the board
def isInside(x, y, N):
    if (x >= 1 and x <= N and
        y >= 1 and y <= N):
        return True
    return False

# Method returns minimum step to reach
# target position
def minStepToReachTarget(knightpos,
                         targetpos, N):

    # all possible movments for the knight
    dx = [2, 2, -2, -2, 1, 1, -1, -1]
    dy = [1, -1, 1, -1, 2, -2, 2, -2]

    queue = []

    # push starting position of knight
    # with 0 distance
    queue.append(cell(knightpos[0], knightpos[1], 0))

    # make all cell unvisited
    visited = [[False for i in range(N + 1)]
                      for j in range(N + 1)]

    # visit starting state
    visited[knightpos[0]][knightpos[1]] = True

    # loop until we have one element in queue
    while(len(queue) > 0):

        t = queue[0]
        queue.pop(0)

        # if current cell is equal to target
        # cell, return its distance
        if(t.x == targetpos[0] and
           t.y == targetpos[1]):
            return t.dist

        # iterate for all reachable states
        for i in range(8):

            x = t.x + dx[i]
            y = t.y + dy[i]

            if(isInside(x, y, N) and not visited[x][y]):
                visited[x][y] = True
                queue.append(cell(x, y, t.dist + 1))

# Driver Code    
if __name__=='__main__':
    N = 30
    knightpos = [1, 1]
    targetpos = [30, 30]
    print(minStepToReachTarget(knightpos,
                               targetpos, N)
---------------------------------------------------------
Author: PrinciRaj1992
URL: https://www.geeksforgeeks.org/minimum-steps-reach-target-knight-set-2/
Code: 
dp = [[0 for i in range(8)] for j in range(8)];


def getsteps(x, y, tx, ty):

    # if knight is on the target
    # position return 0.
    if (x == tx and y == ty):
        return dp[0][0];

    # if already calculated then return
    # that value. Taking absolute difference.
    elif(dp[abs(x - tx)][abs(y - ty)] != 0):
        return dp[abs(x - tx)][abs(y - ty)];
    else:

        # there will be two distinct positions
        # from the knight towards a target.
        # if the target is in same row or column
        # as of knight than there can be four
        # positions towards the target but in that
        # two would be the same and the other two
        # would be the same.
        x1, y1, x2, y2 = 0, 0, 0, 0;

        # (x1, y1) and (x2, y2) are two positions.
        # these can be different according to situation.
        # From position of knight, the chess board can be
        # divided into four blocks i.e.. N-E, E-S, S-W, W-N .
        if (x <= tx):
            if (y <= ty):
                x1 = x + 2;
                y1 = y + 1;
                x2 = x + 1;
                y2 = y + 2;
            else:
                x1 = x + 2;
                y1 = y - 1;
                x2 = x + 1;
                y2 = y - 2;

        elif (y <= ty):
            x1 = x - 2;
            y1 = y + 1;
            x2 = x - 1;
            y2 = y + 2;
        else:
            x1 = x - 2;
            y1 = y - 1;
            x2 = x - 1;
            y2 = y - 2;

        # ans will be, 1 + minimum of steps
        # required from (x1, y1) and (x2, y2).
        dp[abs(x - tx)][abs(y - ty)] = \
        min(getsteps(x1, y1, tx, ty),
        getsteps(x2, y2, tx, ty)) + 1;

        # exchanging the coordinates x with y of both
        # knight and target will result in same ans.
        dp[abs(y - ty)][abs(x - tx)] = \
        dp[abs(x - tx)][abs(y - ty)];
        return dp[abs(x - tx)][abs(y - ty)];

# Driver Code
if __name__ == '__main__':

    # size of chess board n*n
    n = 100;

    # (x, y) coordinate of the knight.
    # (tx, ty) coordinate of the target position.
    x = 4;
    y = 5;
    tx = 1;
    ty = 1;

    # (Exception) these are the four corner points
    # for which the minimum steps is 4.
    if ((x == 1 and y == 1 and tx == 2 and ty == 2) or
            (x == 2 and y == 2 and tx == 1 and ty == 1)):
        ans = 4;
    elif ((x == 1 and y == n and tx == 2 and ty == n - 1) or
        (x == 2 and y == n - 1 and tx == 1 and ty == n)):
        ans = 4;
    elif ((x == n and y == 1 and tx == n - 1 and ty == 2) or
        (x == n - 1 and y == 2 and tx == n and ty == 1)):
        ans = 4;
    elif ((x == n and y == n and tx == n - 1 and ty == n - 1)
        or (x == n - 1 and y == n - 1 and tx == n and ty == n)):
        ans = 4;
    else:

        # dp[a][b], here a, b is the difference of
        # x & tx and y & ty respectively.
        dp[1][0] = 3;
        dp[0][1] = 3;
        dp[1][1] = 2;
        dp[2][0] = 2;
        dp[0][2] = 2;
        dp[2][1] = 1;
        dp[1][2] = 1;

        ans = getsteps(x, y, tx, ty);

    print(ans);
----------------------------------------------------------
Author: _xavier_
Date: October 7, 2021
URL: https://leetcode.com/problems/word-search/discuss/529320/two-easy-python-solution-with-explanation
Code: 
def exist(self, board, word):

        'return true if word is empty

        if not word: 
            return True

        # return false if board is empty

        if not board:
            return False

        m, n, w = len(board), len(board[0]), len(word) - 1

        # inside board, i,  j represent the coordination of current cell to be checked (i meaning row, 
        j meaning column) against the kth character of the 'word' 

        def dfs(i, j, k):

            # backtrack if i or j hit beyond the edge(s) of the 'board'

            if i < 0 or i >= m or j < 0 or j >= n:
                return False

            # backtrack if the cell has been already visited

            if board[i][j] == '#':
                return False

            # backtrack if the cell does not match the current character of the 'word'

            if board[i][j] != word[k]:
                return False        

            # now that we have reached here, it means the cell matches the current character of the 'word'. 
            # just check if it was the last successful checking required (k==w). 
            # if so, we are done: the 'word' matches completely, so return True

            if k == w:
                return True

            # if still not the last character then...
            # save the cell in case we need to backtrack later.  mark the cell as '#' (meaning visited)

            tmp = board[i][j] 
            board[i][j] = '#'

            # so far so good up to the kth character of the 'word' (meaning everything matched up to the kth character)
            # push the pointer one step forward (in order for the next character of the 'word' to be checked shortly)

            k += 1

            # inside for clause below: continue with up, down, left and right of the current cell: 
            # as soon as a match is found out, happily return true (meaning from that point on to the end of 'word', 
            # everything was matched up (due to the dfs), Yay...)

            for x, y in ((-1, 0), (+1, 0), (0, -1), (0, +1)):
                if dfs(i + x, j + y, k):
                    return True

            # we have reached here: meaning none of 4 potential paths (inside the for clause above) got
             matched up to the end. 
            # meaning the current cell is not a good candidate, 
            # so return it to the non-visited pool (by changing back to its original backed-up value 'tmp'). 
            then return False

            board[i][j] = tmp
            return False


        # check the entire board, cell by cell as the starting point. 
        # the value '0' below means we will start off by checking if board[i][j] = word[0]. 
        # inside the dfs function, we push k (one step at a time) which represents the pointer to the current 
        character (of the 'word') required to be checked.

        for i in range(m):
            for j in range(n):
                if dfs(i, j, 0):
                    return True

        # nothing matched successfully from any starting point in the 'board', so we are done, return False
        return False
-------------------------------------
Code: 
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        def does_exist(k, r, c, s):
            if (r, c) in s: return False
            if not (0 <= r < row and 0 <= c < col): return False
            if not k: return board[r][c] == word[k]
            return k and board[r][c] == word[k] and any(map(lambda x: does_exist(k - 1, *x, s|{(r, c)}), ((r, c+1) ,(r, c-1), (r+1, c), (r-1, c))))

        seen, wl, row, col = set(), len(word), len(board), len(board[0])
        return any(does_exist(wl - 1, r, c, seen) for c in range(col) for r in range(row))
------------------------------------------------------------------------
Author: Unknown
URL: https://www.geeksforgeeks.org/minimum-cost-path-left-right-bottom-moves-allowed/?ref=lbp
Code:
using namespace std;

#define ROW 5
#define COL 5

// structure for information of each cell
struct cell
{
    int x, y;
    int distance;
    cell(int x, int y, int distance) :
        x(x), y(y), distance(distance) {}
};

// Utility method for comparing two cells
bool operator<(const cell& a, const cell& b)
{
    if (a.distance == b.distance)
    {
        if (a.x != b.x)
            return (a.x < b.x);
        else
            return (a.y < b.y);
    }
    return (a.distance < b.distance);
}

// Utility method to check whether a point is
// inside the grid or not
bool isInsideGrid(int i, int j)
{
    return (i >= 0 && i < ROW && j >= 0 && j < COL);
}

// Method returns minimum cost to reach bottom
// right from top left
int shortest(int grid[ROW][COL], int row, int col)
{
    int dis[row][col];

    // initializing distance array by INT_MAX
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            dis[i][j] = INT_MAX;

    // direction arrays for simplification of getting
    // neighbour
    int dx[] = {-1, 0, 1, 0};
    int dy[] = {0, 1, 0, -1};

    set<cell> st;

    // insert (0, 0) cell with 0 distance
    st.insert(cell(0, 0, 0));

    // initialize distance of (0, 0) with its grid value
    dis[0][0] = grid[0][0];

    // loop for standard dijkstra's algorithm
    while (!st.empty())
    {
        // get the cell with minimum distance and delete
        // it from the set
        cell k = *st.begin();
        st.erase(st.begin());

        // looping through all neighbours
        for (int i = 0; i < 4; i++)
        {
            int x = k.x + dx[i];
            int y = k.y + dy[i];

            // if not inside boundary, ignore them
            if (!isInsideGrid(x, y))
                continue;

            // If distance from current cell is smaller, then
            // update distance of neighbour cell
            if (dis[x][y] > dis[k.x][k.y] + grid[x][y])
            {
                // If cell is already there in set, then
                // remove its previous entry
                if (dis[x][y] != INT_MAX)
                    st.erase(st.find(cell(x, y, dis[x][y])));

                // update the distance and insert new updated
                // cell in set
                dis[x][y] = dis[k.x][k.y] + grid[x][y];
                st.insert(cell(x, y, dis[x][y]));
            }
        }
    }

    // uncomment below code to print distance
    // of each cell from (0, 0)
    /*
    for (int i = 0; i < row; i++, cout << endl)
        for (int j = 0; j < col; j++)
            cout << dis[i][j] << " ";
    */
    // dis[row - 1][col - 1] will represent final
    // distance of bottom right cell from top left cell
    return dis[row - 1][col - 1];
}

// Driver code to test above methods
int main()
{
    int grid[ROW][COL] =
    {
        31, 100, 65, 12, 18,
        10, 13, 47, 157, 6,
        100, 113, 174, 11, 33,
        88, 124, 41, 20, 140,
        99, 32, 111, 41, 20
    };

    cout << shortest(grid, ROW, COL) << endl;
    return 0;
}
-------------------------------------------------------------------
Author: Neelam Yadav
Date: Jan 30, 2022
URL: https://www.geeksforgeeks.org/depth-first-search-or-dfs-for-a-graph/
Code: 
from collections import defaultdict

# This class represents a directed graph using
# adjacency list representation


class Graph:

    # Constructor
    def __init__(self):

        # default dictionary to store graph
        self.graph = defaultdict(list)

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    # A function used by DFS
    def DFSUtil(self, v, visited):

        # Mark the current node as visited
        # and print it
        visited.add(v)
        print(v, end=' ')

        # Recur for all the vertices
        # adjacent to this vertex
        for neighbour in self.graph[v]:
            if neighbour not in visited:
                self.DFSUtil(neighbour, visited)

    # The function to do DFS traversal. It uses
    # recursive DFSUtil()
    def DFS(self, v):

        # Create a set to store visited vertices
        visited = set()

        # Call the recursive helper function
        # to print DFS traversal
        self.DFSUtil(v, visited)

# Driver code


# Create a graph given
# in the above diagram
g = Graph()
g.addEdge(0, 1)
g.addEdge(0, 2)
g.addEdge(1, 2)
g.addEdge(2, 0)
g.addEdge(2, 3)
g.addEdge(3, 3)

print("Following is DFS from (starting from vertex 2)")
g.DFS(2)
-------------------------------------------------------------
Author: Geeks for Geeks - Kevin Joshi
Date: Aug. 19, 2021
URL: https://www.geeksforgeeks.org/8-puzzle-problem-using-branch-and-bound/
Code: 
import copy

# Importing the heap functions from python
# library for Priority Queue
from heapq import heappush, heappop

# This variable can be changed to change
# the program from 8 puzzle(n=3) to 15
# puzzle(n=4) to 24 puzzle(n=5)...
n = 3

# bottom, left, top, right
row = [ 1, 0, -1, 0 ]
col = [ 0, -1, 0, 1 ]

# A class for Priority Queue
class priorityQueue:

    # Constructor to initialize a
    # Priority Queue
    def __init__(self):
        self.heap = []

    # Inserts a new key 'k'
    def push(self, k):
        heappush(self.heap, k)

    # Method to remove minimum element
    # from Priority Queue
    def pop(self):
        return heappop(self.heap)

    # Method to know if the Queue is empty
    def empty(self):
        if not self.heap:
            return True
        else:
            return False

# Node structure
class node:

    def __init__(self, parent, mat, empty_tile_pos,
                 cost, level):

        # Stores the parent node of the
        # current node helps in tracing
        # path when the answer is found
        self.parent = parent

        # Stores the matrix
        self.mat = mat

        # Stores the position at which the
        # empty space tile exists in the matrix
        self.empty_tile_pos = empty_tile_pos

        # Storesthe number of misplaced tiles
        self.cost = cost

        # Stores the number of moves so far
        self.level = level

    # This method is defined so that the
    # priority queue is formed based on
    # the cost variable of the objects
    def __lt__(self, nxt):
        return self.cost < nxt.cost

# Function to calculate the number of
# misplaced tiles ie. number of non-blank
# tiles not in their goal position
def calculateCost(mat, final) -> int:

    count = 0
    for i in range(n):
        for j in range(n):
            if ((mat[i][j]) and
                (mat[i][j] != final[i][j])):
                count += 1

    return count

def newNode(mat, empty_tile_pos, new_empty_tile_pos,
            level, parent, final) -> node:

    # Copy data from parent matrix to current matrix
    new_mat = copy.deepcopy(mat)

    # Move tile by 1 position
    x1 = empty_tile_pos[0]
    y1 = empty_tile_pos[1]
    x2 = new_empty_tile_pos[0]
    y2 = new_empty_tile_pos[1]
    new_mat[x1][y1], new_mat[x2][y2] = new_mat[x2][y2], new_mat[x1][y1]

    # Set number of misplaced tiles
    cost = calculateCost(new_mat, final)

    new_node = node(parent, new_mat, new_empty_tile_pos,
                    cost, level)
    return new_node

# Function to print the N x N matrix
def printMatrix(mat):

    for i in range(n):
        for j in range(n):
            print("%d " % (mat[i][j]), end = " ")

        print()

# Function to check if (x, y) is a valid
# matrix coordinate
def isSafe(x, y):

    return x >= 0 and x < n and y >= 0 and y < n

# Print path from root node to destination node
def printPath(root):

    if root == None:
        return

    printPath(root.parent)
    printMatrix(root.mat)
    print()

# Function to solve N*N - 1 puzzle algorithm
# using Branch and Bound. empty_tile_pos is
# the blank tile position in the initial state.
def solve(initial, empty_tile_pos, final):

    # Create a priority queue to store live
    # nodes of search tree
    pq = priorityQueue()

    # Create the root node
    cost = calculateCost(initial, final)
    root = node(None, initial,
                empty_tile_pos, cost, 0)

    # Add root to list of live nodes
    pq.push(root)

    # Finds a live node with least cost,
    # add its children to list of live
    # nodes and finally deletes it from
    # the list.
    while not pq.empty():

        # Find a live node with least estimated
        # cost and delete it form the list of
        # live nodes
        minimum = pq.pop()

        # If minimum is the answer node
        if minimum.cost == 0:

            # Print the path from root to
            # destination;
            printPath(minimum)
            return

        # Generate all possible children
        for i in range(n):
            new_tile_pos = [
                minimum.empty_tile_pos[0] + row[i],
                minimum.empty_tile_pos[1] + col[i], ]

            if isSafe(new_tile_pos[0], new_tile_pos[1]):

                # Create a child node
                child = newNode(minimum.mat,
                                minimum.empty_tile_pos,
                                new_tile_pos,
                                minimum.level + 1,
                                minimum, final,)

                # Add child to list of live nodes
                pq.push(child)

# Driver Code

# Initial configuration
# Value 0 is used for empty space
initial = [ [ 1, 2, 3 ],
            [ 5, 6, 0 ],
            [ 7, 8, 4 ] ]

# Solvable Final configuration
# Value 0 is used for empty space
final = [ [ 1, 2, 3 ],
          [ 5, 8, 6 ],
          [ 0, 7, 4 ] ]

# Blank tile coordinates in
# initial configuration
empty_tile_pos = [ 1, 2 ]

# Function call to solve the puzzle
solve(initial, empty_tile_pos, final)
-----------------------------------------------------------------
Author: hemantraj712
Date: Feb 7, 2022
URL: https://www.geeksforgeeks.org/unique-paths-in-a-grid-with-obstacles/
Code: 
def uniquePathsWithObstacles(A):

    # create a 2D-matrix and initializing with value 0
    paths = [[0]*len(A[0]) for i in A]

    # initializing the left corner if no obstacle there
    if A[0][0] == 0:
        paths[0][0] = 1

    # initializing first column of the 2D matrix
    for i in range(1, len(A)):

        # If not obstacle
        if A[i][0] == 0:
            paths[i][0] = paths[i-1][0]

    # initializing first row of the 2D matrix
    for j in range(1, len(A[0])):

        # If not obstacle
        if A[0][j] == 0:
            paths[0][j] = paths[0][j-1]

    for i in range(1, len(A)):
        for j in range(1, len(A[0])):

            # If current cell is not obstacle
            if A[i][j] == 0:
                paths[i][j] = paths[i-1][j] + paths[i][j-1]

    # returning the corner value of the matrix
    return paths[-1][-1]


# Driver Code
A = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
print(uniquePathsWithObstacles(A))
------------------------------------
Code:

'''