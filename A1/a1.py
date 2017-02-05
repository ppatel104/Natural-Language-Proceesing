# coding: utf-8
"""CS585: Assignment 1
This assignment has two parts:
(1) Finite-State Automata
- Implement a function run_fsa that takes in a *deterministic* FSA and
  a list of symbols and returns True if the FSA accepts the string.
- Design a deterministic FSA to represent a toy example of person names.
(2) Parsing
- Read a context-free grammar from a string.
- Implement a function that takes in a parse tree, a sentence, and a
  grammar, and returns True if the parse tree is a valid parse of the
  sentence.
See the TODO comments below to complete your assignment. I strongly
recommend working one function at a time, using a1_test to check that
your implementation is correct.
For example, after implementing the run_fsa function, run `python
a1_test.py TestA1.test_baa_fsa` to make sure your FSA implementation
works for the sheep example. Then, move on to the next function.
"""


#######
# FSA #
#######

def run_fsa(states, initial_state, accept_states, transition, input_symbols):
    """
    Implement a deterministic finite-state automata.
    See test_baa_fsa.
    Params:
      states..........list of ints, one per state
      initial_state...int for the starting state
      accept_states...List of ints for the accept states
      transition......dict of dicts representing the transition function
      input_symbols...list of strings representing the input string
    Returns:
      True if this FSA accepts the input string; False otherwise.
    """
    ###TODO

    #Comments:
    # set state to the initial_state
    # loop through the individual input symbols
    # check if the input_symbol is in given transition state
    # 1) based on the state and an input symbol made transition, and save the "new state" in state
    # 2) if state is in accept state and all the input symbols have been read, then return true.
    # 3) if state is in accept state, and still there are some symbols left, then return false.

    input_length = len(input_symbols)
    state = initial_state
    for i in range(0,input_length):

        if(input_symbols[i] in transition[state]):
            state = transition[state][input_symbols[i]]
            if((state in accept_states) and (i==input_length-1)):
                return True
            elif((state in accept_states) and (i!=input_length-1)):
                return False
        else:
            return False



def get_name_fsa():
    """
    Define a deterministic finite-state machine to recognize a small set of person names, such as:
      Mr. Frank Michael Lewis
      Ms. Flo Lutz
      Frank Micael Lewis
      Flo Lutz
    See test_name_fsa for examples.
    Names have the following:
    - an optional prefix in the set {'Mr.', 'Ms.'}
    - a required first name in the set {'Frank', 'Flo'}
    - an optional middle name in the set {'Michael', 'Maggie'}
    - a required last name in the set {'Lewis', 'Lutz'}
    Returns:
      A 4-tuple of variables, in this order:
      states..........list of ints, one per state
      initial_state...int for the starting state
      accept_states...List of ints for the accept states
      transition......dict of dicts representing the transition function
    """
    ###TODO

    states = [0,1,2,3,4]
    initial_state = 0
    accept_states = [4]
    #based on the name format create an individual transition directory.
    transition0 = {'Mr.':1,'Ms.':1,'Frank':2,'Flo':2}
    transition1 = {'Frank':2,'Flo':2}
    transition2 = {'Michael':3,'Maggie':3,'Lewis':4,'Lutz':4}
    transition3 = {'Lewis':4,'Lutz':4}
    #create a transition directory which contains all the state and transition related to it.
    transition = {0:transition0,
                  1:transition1,
                  2:transition2,
                  3:transition3}
    return states,initial_state,accept_states,transition


###########
# PARSING #
###########

def read_grammar(lines):
    """Read a list of strings representing CFG rules. E.g., the string
    'S :- NP VP'
    should be parsed into a tuple
    ('S', ['NP', 'VP'])
    Note that the first element of the tuple is a string ('S') for the
    left-hand-side of the rule, and the second element is a list of
    strings (['NP', 'VP']) for the right-hand-side of the rule.

    See test_read_grammar.
    Params:
      lines...A list of strings, one per rule
    Returns:
      A list of (LHS, RHS) tuples, one per rule.
    """
    ###TODO

    # Comments:
    # 1) split the individial strings based on :-
    # 2) assign first part of the string to LHS
    # 3) assign the second part of the string to RHS
    # 4) remove all whitespaces in the LHS part.

    Grammer_rules = list()
    for sent in lines:
        string = sent.split(':-')
        LHS = string[0]
        RHS = string[1]
        RHS = RHS.split()
        rule = (LHS.replace(" ",""),RHS)
        Grammer_rules.append(rule)
    return Grammer_rules


class Tree:
    """A partial implementation of a Tree class to represent a parse tree.
    Each node in the Tree is also a Tree.
    Each Tree has two attributes:
      - label......a string representing the node (e.g., 'S', 'NP', 'dog')
      - children...a (possibly empty) list of children of this
                   node. Each element of this list is another Tree.
    A leaf node is a Tree with an empty list of children.
    """

    def __init__(self, label, children=[]):
        """The constructor.
        Params:
          label......A string representing this node
          children...An optional list of Tree nodes, representing the
                     children of this node.
        This is done for you and should not be modified.
        """
        self.label = label
        self.children = children

    def __str__(self):
        """
        Print a string representation of the tree, for debugging.
        This is done for you and should not be modified.
        """
        s = self.label
        for c in self.children:
            s += ' ( ' + str(c) + ' ) '
        return s

    def get_leaves(self):
        """
        Returns:
          A list of strings representing the leaves of this tree.
        See test_get_leaves.
        """
        ###TODO

        # Comments:
        # Recursive function to get childs of a given node(which itself a tree) in a tree.
        # if the the node(or a tree) doesn't have a children then it's a leaf, so it's label and children are both same.

        childs = []
        def get_childs(self):
            tree_1 = Tree(self.label,self.children)
            if(not tree_1.children):
                childs.append(tree_1.label)
            for i in range(0,len(self.children)):
                get_childs(self.children[i])
            return childs
        get_childs(self)
        return childs



    def get_productions(self):
        """Returns:
          A list of tuples representing a depth-first traversal of
          this tree.  Each tuple is of the form (LHS, RHS), where LHS
          is a string representing the left-hand-side of the
          production, and RHS is a list of strings representing the
          right-hand-side of the production.
        See test_get_productions.
        """
        ###TODO

        # Comments:
        # recursive function get_production(tree) to get productions of a tree.
        # If a tree doesn't have any children then it's false.
        # Given a tree, it will assign LHS of a production a label of it, and for RHS it will assign labels of it's successive childrens.
        # loop through each childrens of a tree and call the function recursively.

        productions = []
        def get_production(self):
            RHS = []
            tree_1 = Tree(self.label,self.children)
            if(not tree_1.children):
                return False
            LHS = self.label
            for i in range(0,len(self.children)):
                tree_2 = self.children[i]
                RHS.append(tree_2.label)
            productions.append((LHS,RHS))
            for i in range(0,len(self.children)):
                get_production(self.children[i])
        get_production(self)
        return productions


def is_pos(rule, rules):
    """
    Returns:
      True if this rule is a part-of-speech rule, which is true if none of the
      RHS symbols appear as LHS symbols in other rules.
    E.g., if the grammar is:
    S :- NP VP
    NP :- N
    VP :- V
    N :- dog cat
    V :- run likes
    Then the final two rules are POS rules.
    See test_is_pos.
    This function should be used by the is_valid_production function
    below.
    """
    ###TODO

    # Comments:
    # Assign RHS part of a given rule to RHS.
    # Collect LHS part of each rules in a list called LHS.
    # loop thorugh the each RHS symbol and check whether it is in list LHS or not.
    # if RHS is in LHS, return False. Else, return True.

    RHS = rule[1]
    LHS = []
    for i in range(0,len(rules)):
        LHS.append(rules[i][0])
    for i in range(0,len(RHS)):
        if (RHS[i] in LHS):
            return False
        else:
            return True



def is_valid_production(production, rules):
    """
    Params:
      production...A (LHS, RHS) tuple representing one production,
                   where LHS is a string and RHS is a list of strings.
      rules........A list of tuples representing the rules of the grammar.
    Returns:
      True if this production is valid according to the rules of the
      grammar; False otherwise.
    See test_is_valid_production.
    This function should be used in the is_valid_tree method below.
    """
    ###TODO

    # Comments:
    # 1) loop thorugh all the rules, and check which rule's LHS matches with the given production's LHS.
    # 2) For that rule, check whether it is a part of speech rule or not using is_pos() function.
    # 3) If the rule isn't a POS then production's RHS should be exactly match the RHS of the rule, else it is False.
    # 4) If the rule isn't a POS then check if the production RHS's first symbol is in that rule's RHS, if yes then it's True.


    for i in range(0,len(rules)):
        if((production[0]==rules[i][0])):
            if(is_pos(rules[i],rules)):
                if(production[1][0] in rules[i][1]):
                    return True
            else:
                if(production[1]==rules[i][1]):
                    return True
    return False



def is_valid_tree(tree, rules, words):
    """
    Params:
      tree....A Tree object representing a parse tree.
      rules...The list of rules in the grammar.
      words...A list of strings representing the sentence to be parsed.
    Returns:
      True if the tree is a valid parse of this sentence. This requires:
        - every production in the tree is valid (present in the list of rules).
        - the leaf nodes in the tree match the words in the sentence, in order.
    See test_is_valid_tree.
    """
    ###TODO

    # Comments:
    # 1) get childs of a given tree using the get_leaves() function.
    # 2) get productions of a given tree using the get_productions() function.
    # 3) If the childrens of a given tree doesn't exactly match the words in a given sentence, then it's False.
    # 4) Otherwise, loop through all the productions, and if any one of the productions of a Tree isn't correct, then it's False.
    
    childs = tree.get_leaves()
    productions = tree.get_productions()
    if (words!=childs):
        return False
    for i in range(0,len(productions)):
        if (is_valid_production(productions[i],rules)!=True):
            return False
    return True
