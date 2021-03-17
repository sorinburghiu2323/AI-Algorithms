import random
from math import sqrt

DEVELOPMENT = False  # Set this to True for testing purposes to not have to input.
IS_EUCLIDEAN = True  # Set to False for Manhattan heuristics or True for Euclidean.
DEVELOPMENT_START = [[7, 2, 4], [5, 0, 6], [8, 3, 1]]
DEVELOPMENT_GOAL = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


class Node:
    """
    Node class to represent a state for the 8-puzzle.
    """

    def __init__(self, data, g_value, goal, euclidean=True, parent=None):
        self.data = data
        self.g_value = g_value
        self.parent = parent
        self.f_value = self.calculate_f(goal, euclidean)

    def calculate_f(self, goal, euclidean):
        """
        Calculate the f value: f(n) = h(n) + g(n)
        :param goal: goal state.
        :param euclidean: True if to use Euclidean heuristics.
        :return: f value for this state.
        """
        return self.calculate_h(self.data, goal, euclidean) + self.g_value

    def calculate_h(self, initial, goal, euclidean):
        """
        Calculate the h value using either:
            - Euclidean distance.
            - Manhattan distance.
        :param initial: initial state.
        :param goal: final state.
        :param euclidean: True if to use Euclidean heuristics.
        :return: h value for this state.
        """
        temp = 0
        for i in range(3):
            for j in range(3):
                goal_pos = self.find_tile_coordinates(goal, initial[i][j])
                if euclidean:
                    temp += sqrt((i - goal_pos[0])**2 + (j - goal_pos[1])**2)
                else:
                    temp += abs(i - goal_pos[0]) + abs(j - goal_pos[1])
        return temp

    def find_tile_coordinates(self, state, tile):
        """
        Find coordinates of a given tile.
        :param state: state of matrix.
        :param tile: tile number to be found (e.g: 2)
        :return: [x, y] coordinates.
        """
        for i in range(3):
            for j in range(3):
                if state[i][j] == tile:
                    return [i, j]

    def generate_children(self, parent, goal, euclidean=True):
        """
        Generate children of the node.
        :param goal: goal state.
        :param euclidean: True if to use Euclidean heuristics.
        :return: list of child Nodes.
        """
        tile_coords = self.find_tile_coordinates(self.data, 0)
        x, y = tile_coords[0], tile_coords[1]
        val_list = [[x, y - 1], [x, y + 1], [x - 1, y], [x + 1, y]]
        children = []
        for i in val_list:
            child = self.shuffle(x, y, i[0], i[1])
            if child is not None:
                child_node = Node(child, self.g_value + 1, goal, euclidean, parent)
                children.append(child_node)
        return children

    def shuffle(self, x1, y1, x2, y2):
        """
        Move the empty cell to the given position, return None if it
        is not a possible action.
        :param x1: Initial x position.
        :param y1: Initial y position.
        :param x2: Goal x position.
        :param y2: Goal y position.
        :return: state puzzle or None.
        """
        if 0 <= x2 < len(self.data) and 0 <= y2 < len(self.data):
            temp_puzzle = self.copy(self.data)
            temp = temp_puzzle[x2][y2]
            temp_puzzle[x2][y2] = temp_puzzle[x1][y1]
            temp_puzzle[x1][y1] = temp
            return temp_puzzle
        else:
            return None

    def copy(self, state):
        """
        Copy the current state into a new temporary one.
        :param state: state to copy.
        :return: new state.
        """
        temp = []
        for i in state:
            t = []
            for j in i:
                t.append(j)
            temp.append(t)
        return temp


class Puzzle:
    """
    Main class to run the 8-puzzle game.
    """

    def __init__(self, initial: Node):
        self.open = [initial]
        self.closed = []
        self.iteration_count = 0

    def run(self, final, euclidean):
        """
        Run the puzzle. This is the main loop for the A* algorithm.
        :param final: end state as matrix.
        :param euclidean: True if to use Euclidean heuristics.
        :return: None.
        """
        while self.open:

            current = self.lowest()
            self.iteration_count += 1

            # If end state was reached, end.
            if current.data == final:
                self.end(current)
                print("\nMoves: ", current.g_value)
                print("Iterations: ", self.iteration_count)
                break

            # Remove current node from open list and add it to closed list.
            self.open.remove(current)
            self.closed.append(current)
            self.loading()

            # Loop through the node children and handle them adequately.
            neighbours = current.generate_children(current, final, euclidean)
            for neighbour in neighbours:
                exist_in_open = self.is_in_open(neighbour)
                if exist_in_open:
                    if exist_in_open.g_value < neighbour.g_value:
                        continue
                exist_in_closed = self.is_in_closed(neighbour)
                if exist_in_closed:
                    if exist_in_closed.g_value < neighbour.g_value:
                        continue
                self.open.append(neighbour)

    def loading(self):
        """
        Quality of life loading messages to keep the user entertained.
        :return: None.
        """
        if self.iteration_count % 2000 == 0:
            loading_statements = [
                "Going really deep into the tree...",
                "Searching deeply...",
                "Analysing open nodes...",
                "Loading...",
                "Optimising path...",
            ]
            print(random.choice(loading_statements))

    def end(self, current):
        """
        End function to print the sequence.
        :param current: current node (last one).
        :return: None.
        """
        move_list = []
        while current.parent:
            move_list.append(current)
            current = current.parent
        move_list.append(current)
        for move in move_list[::-1]:
            print("\n <<<<<< \n")
            self.pretty_print(move)

    def lowest(self):
        """
        Find lowest element in the open list.
        :return: Node with lowest f value.
        """
        lowest = self.open[0]
        for node in self.open:
            if node.f_value < lowest.f_value:
                lowest = node
        return lowest

    def is_in_open(self, node):
        """
        Check if node is in open list.
        :param node: Node to be checked.
        :return: the copy Node else None.
        """
        for n in self.open:
            if n.data == node.data:
                return n
        return None

    def is_in_closed(self, node):
        """
        Check if node is in closed list.
        :param node: Node to be checked.
        :return: the copy Node else None.
        """
        for n in self.closed:
            if n.data == node.data:
                return n
        return None

    def pretty_print(self, node):
        """
        Print a node in a nice format.
        :param node: Node to be printed.
        :return: None.
        """
        for row in node.data:
            pretty = " "
            for i in row:
                pretty += str(i) + " "
            print("|" + pretty + "|")


def initialize_input():
    """
    Generate input for a start and end state.
    :return: e.g. [[1,2,3],[4,5,6],[7,8,9]]
    """
    matrix = []
    for i in range(3):
        temp = input().split(" ")
        row = []
        for j in temp:
            row.append(int(j))
        matrix.append(row)
    return matrix


if __name__ == '__main__':

    # Get inputs if not in development, otherwise use default values.
    if not DEVELOPMENT:
        print("Input each number separated by a space, enter new row by pressing enter.")
        print("Please input the start state:")
        start = initialize_input()
        print("Please input the goal state:")
        end = initialize_input()
        choice = input("Enter 1 for Euclidean heuristic or 2 for Manhattan heuristic:")
        if choice == "2":
            is_euclidean = False
        else:
            is_euclidean = True
    else:
        start = DEVELOPMENT_START
        end = DEVELOPMENT_GOAL
        is_euclidean = IS_EUCLIDEAN

    # Run A star search for given problem.
    print("Calculating optimal path...")
    start_node = Node(start, 0, end, is_euclidean)
    puzzle = Puzzle(start_node)
    puzzle.run(end, is_euclidean)
