import numpy as np
import pyspiel
from open_spiel.python.algorithms.mcts import SearchNode

from alpha_one.alg.imperfect_information import ImperfectInformationMCTSEvaluator
from alpha_one.game.information_set import InformationSetGenerator, PUBLIC_OBSERVER_PLAYER_ID
from alpha_one.game.observer import OmniscientObserver


class ImperfectInformationSearchNode(SearchNode):
    """
    A node in a search tree that can represent one of the following:
        - Game Tree Node:
            Once a player has guessed a state he lands in the game tree. The attributes of a game tree node have
            following meaning:
                * action: the guess action that lead to this game tree node
                * player: the player that guessed (will also be the same player that should choose an action from here on)
                * state: the state that was guessed
        - Observation Tree Node:
            Once a player has chosen an action, an observation will be made automatically. The observation tree node
            represents the state in the game directly after making this observation. In this case, the attributes
            have the following meaning:
                * action: the action (potentially of the other player) that caused this observation
                * player: the player who played the action
                * state: the state of the game after the player applied its action
                * deciding_player: the player who will have to guess the game state now
                * information_set_generator: capsules the different observations of the players after the action was applied
                    and holds the belief states of both players about the true game state

    Furthermore, there are 2 trees that are simultaneously explored:
        - Opponent Model tree:
            here, the game is modeled from the opponent's perspective, i.e., he has no access to the root player's
            private information. Exploring this tree is equivalent to reasoning about how the opponent might play
        - Root Player Model tree:
            here, the game is modeled from the root's players perspective, i.e., with access to the root player's
            private information. The goal is to get the actual rewards that the game will yield. This is necessary, as
            the Opponent Model tree will make wrong guesses about the root player's private information which will
            give rewards that won't be realistic from the root player's perspective. The root player model tree tries
            to mimic the actions of the opponent but chooses different actions for his own future game states. This is
            necessary, as when the opponent guesses the root player's cards wrong, then he might model the root player
            to play a card that the root player does not have. As such, the root player model tree and the opponent
            model tree will diverge eventually. This is not too problematic though, as the goal is only to get realistic
            rewards if the game is executed with the private information of the root player in mind.

        The Opponent Model Tree is a pure hypothetical tree that covers many game states that are not possible to
        reach anymore given the private information of the player. It is still important to model these hypothetical
        states. Consider the following example of Kuhn Poker:
            Player 0 private card: [X]  # The actual card does not matter for this thought experiment
            Player 1 private card: [2]
                * Player 0 PASS
                * Player 1 IIG-MCTS ->
                    He will always win, because he has the highest card. If he wants to win more than one point though,
                    player 1 has to BET now. In order to see this, he has to make the following hypothetical reasoning:
                        Player 0 could have card [1], in this case Player 0 could think that Player 1 has card [0].
                        Then, Player 0 would maximize his reward by calling Player 1's BET, as then Player 0 would
                        win 1 vs 0 -> yields [2, -2] reward.
                    Player 1 knows that he has card [2] though in which case the above scenario would yield the exact
                    opposite reward [-2, 2]. As such, Player 1's best guess is to assume that Player 0 has card [1]
                    and thinks that Player 1 has card [0]. This is essentially a bluff, Player 1 downplays his strong
                    [2] and hopes that the other player underestimates his hand. This is the only way that Player 1
                    can gain 2 money here. The hypothetical reasoning of what player 0 might think and do is what the
                    Opponent Model Tree does. The realization that this would yield a different reward than what the
                    opponent would expect is the job of the Root Player Model Tree.

        The above thought experience outlines the meaning of actual_reward and actual_outcome.
        During backup of the rewards, the Opponent Model Tree will receive actual_rewards and actual_outcomes that
        represent realistic rewards of the actual game given the root player's private information. These rewards
        won't always exactly correspond to what happens in the Opponent Model Tree as many of the states explored there
        are not accessible from the current game state as known by the root player. This might even include that
        actions have been played that in sequence are not applicable to the current game state (due to wrongly guessing
        the root player's private information). As such the Root Player Model Tree makes a best attempt to provide
        realistic rewards if the game proceeds with a game state known to be possible by the root player, but when
        the opponent proceeds in the same way as in the Opponent Model Tree.
    """

    def __init__(self, action, player: int, prior, regular_game_tree_node=False, state=None,
                 information_set_generator: InformationSetGenerator = None, terminal_state=None, deciding_player=None):
        super(ImperfectInformationSearchNode, self).__init__(action, player, prior)

        self.regular_game_tree_node = regular_game_tree_node
        self.state = state
        self.information_set_generator = information_set_generator
        self.terminal_state = terminal_state
        self.actual_outcome = None  # Models the outcome you get if you know the private information of a player
        self.actual_reward = 0.0
        self.deciding_player = deciding_player

    @staticmethod
    def from_game_tree_node(state_id: int, player: int, prior, state: pyspiel.State):
        return ImperfectInformationSearchNode(state_id, player, prior, regular_game_tree_node=True,
                                              state=state)

    @staticmethod
    def from_observation_tree_node(information_set_generator: InformationSetGenerator, causing_action, player: int,
                                   prior, terminal_state=None, causing_state=None, deciding_player=None):
        return ImperfectInformationSearchNode(causing_action, player, prior, regular_game_tree_node=False,
                                              information_set_generator=information_set_generator,
                                              terminal_state=terminal_state,
                                              state=causing_state, deciding_player=deciding_player)

    def is_game_tree_node(self):
        return self.regular_game_tree_node

    def is_observation_tree_node(self):
        return not self.regular_game_tree_node

    def is_terminal(self):
        return self.terminal_state is not None

    def get_terminal_state(self):
        return self.terminal_state

    def get_deciding_player(self):
        assert self.is_observation_tree_node(), "Only observation tree nodes (with automatic consequent observation have a deciding player)"
        return self.deciding_player

    def calculate_information_set(self):
        assert self.is_observation_tree_node(), "Can only be called on observation tree nodes"
        self.information_set_generator.calculate_information_set(self.player)

    def investigate(self, level=0):
        assert self.is_observation_tree_node(), "Can only investigate observation nodes"

        for s1 in self.children:
            if len(s1.children) > 0:
                if self.get_deciding_player() >= 0:
                    omniscient_observer = OmniscientObserver(s1.state.get_game())
                    observation = omniscient_observer.get_observation_string(s1.state)
                    # observation = s1.state.observation_string(1 - self.get_deciding_player())
                    print(''.join(['  '] * level),
                          f"Player {self.get_deciding_player()} guess action {observation}, explore: {s1.explore_count}, reward: {s1.total_reward: 0.3f}, actual reward: {s1.actual_reward}, actual_outcome: {s1.actual_outcome}")
                for s2 in s1.children:
                    print(''.join(['  '] * (level + 1)),
                          f"p{s2.player}, {s2.action}, explore: {s2.explore_count}, reward: {s2.total_reward: 0.2f}, actual reward: {s2.actual_reward:0.2f}",

                          np.array(s2.actual_outcome).mean(axis=0) if s2.actual_outcome else s2.actual_outcome,
                          f", puct {self.puct_value(s1.explore_count, 3):0.3f}")
                    s2.investigate(level=level + 2)
            else:
                print(''.join(['  '] * level), 'orphan node', s1)

    def to_str(self, state=None):
        return f"{super(ImperfectInformationSearchNode, self).to_str(state)}, actual reward: {self.actual_reward}"


class ImperfectInformationMCTSBot(pyspiel.Bot):
    """Bot that uses Monte-Carlo Tree Search algorithm."""

    def __init__(self,
                 game: pyspiel.Game,
                 uct_c: float,
                 max_simulations: int,
                 evaluator: ImperfectInformationMCTSEvaluator,
                 optimism: float = 1.0,
                 solve=True,
                 random_state=None,
                 child_selection_fn=SearchNode.puct_value,
                 dirichlet_noise=None,
                 verbose=False):
        """Initializes an IIG-MCTS Search algorithm in the form of a bot.

        Args:
          game: A pyspiel.Game to play.
          uct_c: The exploration constant for UCT.
          max_simulations: How many iterations of MCTS to perform. Each simulation
            will result in two calls to the evaluator (one for Opponent Model Tree and for Root Player Model Tree).
          evaluator: A `Evaluator` object to use to evaluate a leaf node.
          optimism: value in (0, 1]. How optimistic the bot is when guessing.
                    1 means the bot is biased towards guessing states that are beneficial for the player
                    small values (close to 0, but not 0!) mean that the bot more or less guessing according to its
                    prior distribution of the information set states.
          solve: Whether to back up solved states.
          random_state: An optional numpy RandomState to make it deterministic.
          child_selection_fn: A function to select the child in the descent phase.
            The default is UCT.
          dirichlet_noise: A tuple of (epsilon, alpha) for adding dirichlet noise to
            the policy at the root. This is from the alpha-zero paper.
          verbose: Whether to print information about the search tree before
            returning the action. Useful for confirming the search is working
            sensibly.

        Raises:
          ValueError: if the game type isn't supported.
        """
        pyspiel.Bot.__init__(self)
        # Check that the game satisfies the conditions for this MCTS implemention.
        game_type = game.get_type()
        if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
            raise ValueError("Game must have terminal rewards.")
        if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
            raise ValueError("Game must have sequential turns.")

        self._game = game
        self.uct_c = uct_c
        self.optimism = optimism
        self.max_simulations = max_simulations
        self.evaluator = evaluator
        self.verbose = verbose
        self.solve = solve  # Currently this
        self.max_utility = game.max_utility()
        self._dirichlet_noise = dirichlet_noise
        self._random_state = random_state or np.random.RandomState()
        self._child_selection_fn = child_selection_fn

    def _apply_tree_policy(self, root_opponent, root):
        """
        Applies the UCT policy to play the game until reaching a leaf node.

        A leaf node is defined as a node that is terminal or has not been evaluated
        yet. If it reaches a node that has been evaluated before but hasn't been
        expanded, then expand it's children and continue.
        In each iteration, it will move one step forward in both the Opponent Model Tree and the Root Player Model Tree.
        It stops once it has reached a leaf node in both trees.

        Parameters
        ----------
            root_opponent:
                The root node in the Opponent Model Tree.
            root:
                The root node in the Root Player Model Tree.

        Returns
        -------
            visit_path_opponent:
                A list of nodes descending from the opponent root node to a leaf node.
            opponent_leaf_node:
                The unexplored node in the Opponent Model Tree.
            visit_path_root:
                A list of nodes descending from the root node to a leaf node.
            root_leaf_node:
                The unexplored node in the Root Player Model Tree
        """
        # Opponent Modeling
        visit_path_opponent = [root_opponent]
        current_node_opponent = root_opponent
        information_set_generator_opponent = root_opponent.information_set_generator

        # Root player modeling
        visit_path_root = [root]
        current_node_root = root
        information_set_generator_root = root.information_set_generator
        root_player_state = None
        root_player = information_set_generator_opponent.current_player()

        if self.verbose:
            print('---')
        # Continue until a leaf node is found in both trees
        while (not current_node_opponent.is_terminal() or not current_node_root.is_terminal()) \
                and (current_node_opponent.explore_count > 0 or current_node_root.explore_count > 0):

            if self.verbose:
                print(f"Visit Path: {[str(node) for node in visit_path_opponent]}")

            # Holds the node being explored if the other tree has already reached a leaf
            running_node = current_node_root if current_node_opponent.is_terminal() else current_node_opponent

            # =========================================================================
            # Game Tree Nodes: Choose an action. Happens directly after a state was guessed
            # =========================================================================

            if running_node.is_game_tree_node():
                if self.verbose:
                    print('game tree node')

                # -------------------------------------------------------------------------
                # Opponent Modeling (Choosing action)
                # -------------------------------------------------------------------------

                if not current_node_opponent.is_terminal():
                    if information_set_generator_opponent.current_player() == -1:
                        current_node_opponent = self._handle_chance_node(current_node_opponent,
                                                                         information_set_generator_opponent,
                                                                         1 - root_player)
                    else:
                        chosen_child, current_node_opponent = self._handle_choose_action_opponent(current_node_opponent,
                                                                                                  information_set_generator_opponent,
                                                                                                  root_opponent)
                    visit_path_opponent.append(current_node_opponent)

                # -------------------------------------------------------------------------
                # Root Player modeling (Choosing action)
                # -------------------------------------------------------------------------

                if not current_node_root.is_terminal():
                    if information_set_generator_root.current_player() == -1:
                        current_node_root = self._handle_chance_node(current_node_root,
                                                                     information_set_generator_root,
                                                                     root_player)
                    else:
                        current_node_root, root_player_state = self._handle_choose_action_root(chosen_child,
                                                                                               current_node_opponent,
                                                                                               current_node_root,
                                                                                               information_set_generator_root,
                                                                                               root, root_player,
                                                                                               visit_path_root)
                    visit_path_root.append(current_node_root)

            # =========================================================================
            # Observation/Guess Tree nodes: a state in the information set has to be guessed
            # =========================================================================

            elif running_node.is_observation_tree_node():
                if self.verbose:
                    print('observation tree node')

                # -------------------------------------------------------------------------
                # Opponent Modeling (Guessing)
                # -------------------------------------------------------------------------

                if not current_node_opponent.is_terminal():
                    information_set_generator_opponent = current_node_opponent.information_set_generator
                    if information_set_generator_opponent.current_player() == -1:
                        current_node_opponent = self._handle_chance_node_guess(current_node_opponent)
                    else:
                        chosen_child, current_node_opponent, information_set_generator_opponent = self._handle_guess_state_opponent(
                            current_node_opponent, information_set_generator_opponent)
                    visit_path_opponent.append(current_node_opponent)

                # -------------------------------------------------------------------------
                # Root Player Modeling (Guessing)
                # -------------------------------------------------------------------------

                if not current_node_root.is_terminal():
                    information_set_generator_root = current_node_root.information_set_generator

                    if information_set_generator_root.current_player() == -1:
                        current_node_root = self._handle_chance_node_guess(current_node_root)
                    else:
                        current_node_root = self._handle_guess_state_root(chosen_child, current_node_opponent,
                                                                          current_node_root,
                                                                          information_set_generator_root, root_player,
                                                                          root_player_state, visit_path_opponent,
                                                                          visit_path_root)

                    visit_path_root.append(current_node_root)

        if self.verbose and current_node_opponent.is_terminal():
            print(current_node_opponent.state.returns(), current_node_root.state.returns(), current_node_opponent.state,
                  current_node_root.state)

        return visit_path_opponent, current_node_opponent, visit_path_root, current_node_root

    # =========================================================================
    # Helper Methods
    # =========================================================================
    # Helper Methods: Guessing State
    # -------------------------------------------------------------------------

    def _handle_guess_state_root(self, chosen_child, current_node_opponent, current_node_root,
                                 information_set_generator_root, root_player, root_player_state, visit_path_opponent,
                                 visit_path_root):
        self._handle_expand_leaf(current_node_root, information_set_generator_root)
        if root_player_state is None:
            assert not current_node_opponent.is_terminal(), f"Cannot run IIG-MCTS on terminal node, {len(visit_path_opponent)}, {len(visit_path_root)}"
            assert chosen_child is not None, f"Opponent has to choose an action in first round"
            # In the beginning, determinize Root and opponent modeling with same state
            corresponding_guess = [c for c in current_node_root.children if
                                   c.action == chosen_child.action]
            assert len(
                corresponding_guess) == 1, f"Could not find corresponding guess {chosen_child.action}"
            current_node_root = corresponding_guess[0]
            root_player_state = current_node_root.state
        else:
            # Root modeling can diverge from opponent modeling by choosing a different action than in
            # the Opponent Model Tree
            chosen_child_root = max(
                current_node_root.children,
                key=lambda c: self._child_selection_fn(  # pylint: disable=g-long-lambda
                    c, current_node_root.explore_count, self.uct_c / self.optimism))
            current_node_root = chosen_child_root
            # root_player_state = current_node_root.state  # TODO: does it matter?
        # Only when root player chooses a state, beliefs are updated.
        if current_node_root.state.current_player() == root_player:
            assert current_node_root.state.current_player() == root_player, f"root player is {root_player}, current state: {current_node_root.state.current_player()}"

            # Update root player's belief with just chosen state
            information_set_generator_root.previous_information_set[
                current_node_root.state.current_player()] = [current_node_root.state.clone()]
            # information_set_generator_root.previous_information_set[
            #     1 - current_node_root.state.current_player()] = [
            #     current_node_root.state.clone()]

            # Opponent is allowed to make wrong guesses, i.e., his beliefs are drawn from public observations
            if information_set_generator_root._public_observer is not None:
                opponent_states = information_set_generator_root.get_plausible_states(
                    1 - root_player,
                    current_node_root.state)
                assert len(
                    opponent_states) > 0, "Opponent's information set will be empty after determinization"
                information_set_generator_root.previous_information_set[
                    1 - current_node_root.state.current_player()] = opponent_states
            else:
                information_set_generator_root.previous_information_set[
                    1 - root_player_state.current_player()] = [
                    root_player_state.clone()]
        # else:
        #     information_set_generator_root = information_set_generator_root.clone()
        #     information_set_generator_root.previous_information_set[current_node_root.state.current_player()] = [
        #         current_node_root.state.clone()]
        #
        #     # Don't change root player's information set (correct game state is there)
        return current_node_root

    def _handle_guess_state_opponent(self, current_node_opponent, information_set_generator_opponent):
        chosen_child = None  # Will be filled for sure in the first iteration
        self._handle_expand_leaf(current_node_opponent, information_set_generator_opponent)
        # Choose node, i.e., guess game state, with largest UCT value
        chosen_child = max(
            current_node_opponent.children,
            key=lambda c: self._child_selection_fn(  # pylint: disable=g-long-lambda
                c, current_node_opponent.explore_count, self.uct_c / self.optimism))
        if self.verbose:
            print(
                f"  - Player {current_node_opponent.player} guesses state {current_node_opponent.children.index(chosen_child)}")
        current_node_opponent = chosen_child  # next node will be guessed game state
        working_state = chosen_child.state
        # Determinize on first guess
        information_set_generator_opponent = information_set_generator_opponent.clone()
        # Update guessing player's belief with just chosen state
        information_set_generator_opponent.previous_information_set[working_state.current_player()] = [
            working_state.clone()]
        # Update opponent's beliefs with public observable states that are ok with the just chosen state
        if information_set_generator_opponent._public_observer is not None:
            opponent_states = information_set_generator_opponent.get_plausible_states(
                1 - working_state.current_player(),
                working_state)
            assert len(
                opponent_states) > 0, "Opponent's information set will be empty after determinization"
            information_set_generator_opponent.previous_information_set[
                1 - working_state.current_player()] = opponent_states
        else:
            information_set_generator_opponent.previous_information_set[
                1 - working_state.current_player()] = [
                working_state.clone()]
        return chosen_child, current_node_opponent, information_set_generator_opponent

    def _handle_expand_leaf(self, current_node, information_set_generator):
        if not current_node.children:
            # Evaluator calculates information set and computes a prior probability for guessing each state
            evaluated_information_set = self.evaluator.prior_observation_node(information_set_generator)
            self._random_state.shuffle(evaluated_information_set)
            assert len(information_set_generator.calculate_information_set()) >= 1, \
                f"No state in information set, observation history: {information_set_generator.get_observation_history()}"
            for state_id, (state, prior) in enumerate(evaluated_information_set):
                game_tree_node = ImperfectInformationSearchNode.from_game_tree_node(
                    state_id,
                    information_set_generator.current_player(),
                    # Player that chooses next action also guesses now from observation
                    prior,
                    state  # .clone()
                )  # TODO: is cloning necessary?
                current_node.children.append(game_tree_node)

    # -------------------------------------------------------------------------
    # Helper Methods: Choose Action
    # -------------------------------------------------------------------------

    def _handle_choose_action_root(self, chosen_child, current_node_opponent, current_node_root,
                                   information_set_generator_root, root, root_player, visit_path_root):
        root_player_state = current_node_root.state  # State that was guessed
        if not current_node_root.children:
            # For a new node, initialize its state, then choose a child as normal.
            # Choose actions with regard to just guessed state
            legal_actions = self.evaluator.prior(root_player_state)
            if current_node_root is root and self._dirichlet_noise:
                epsilon, alpha = self._dirichlet_noise
                noise = self._random_state.dirichlet([alpha] * len(legal_actions))
                legal_actions = [(a, (1 - epsilon) * p + epsilon * n)
                                 for (a, p), n in zip(legal_actions, noise)]
            # Reduce bias from move generation order.
            self._random_state.shuffle(legal_actions)

            player = root_player_state.current_player()  # Player that has to choose an action now
            for action, prior in legal_actions:
                assert len(information_set_generator_root.previous_information_set[
                               root_player]) == 1, "Root should always have exact one information set state"

                # In the Root Player Model Tree, we always apply actions to the root player's guessed state
                # This state always contains the root player's original private information. This is
                # necessary in order to be able to get correct rewards in the end.
                # For the root player, this does not change anything, as he guessed this state anyway.
                # For the opponent player, this does not mean that he may not guess the root player's
                # Private information wrong. In fact, the actions chosen by the opponent player still
                # depend on a potentially wrongly guessed state which is fine and desired (see a few lines
                # above)
                # Essentially, this means that Opponent may guess wrong but observations are made on
                # actual game state
                state_copy = information_set_generator_root.previous_information_set[root_player][
                    0].clone()

                state_copy.apply_action(action)
                if state_copy.is_terminal():
                    # Store state in observation node, Observation is useless as the game is already ended.
                    # This is necessary, as we do not model the resulting state after applying an action
                    # but just the resulting observation that this resulting state induces
                    # To be able to get the returns from terminal state, we include the state in observation
                    # nodes that result from terminal states
                    terminal_state = state_copy
                else:
                    terminal_state = None

                information_set_generator_copy = information_set_generator_root.clone()
                information_set_generator_copy.register(state_copy, action)

                child_node = ImperfectInformationSearchNode.from_observation_tree_node(
                    information_set_generator_copy,
                    action,
                    player,
                    prior,
                    terminal_state=terminal_state,
                    causing_state=state_copy,
                    deciding_player=state_copy.current_player())
                current_node_root.children.append(child_node)
                # This will be the last node to be considered in the Root Player Model Tree
        if len(visit_path_root) == 2:
            # First time the root player plays an action, he should play the same action as
            # when modeling opponent. this allows to directly map the rewards to the root player's first
            # chosen action
            corresponding_action_root = [c for c in current_node_root.children if
                                         c.action == current_node_opponent.action]
            assert len(corresponding_action_root) == 1, \
                f"Could not find corresponding root action {current_node_opponent.action}"
            chosen_child_root = corresponding_action_root[0]
        elif True or root_player_state.current_player() == root_player:
            # TODO: currently, we let the root player model the opponent moves independently instead
            # TODO: of aligning them, as this leads to errors when modeling chance nodes
            # TODO: Always executing this branch causes a failed test
            chosen_child_root = max(
                current_node_root.children,
                key=lambda c: self._child_selection_fn(  # pylint: disable=g-long-lambda
                    c, current_node_root.explore_count, self.uct_c))
        else:
            # TODO: should we really fix the opponent's actions to be the same as when opponent models himself?
            assert chosen_child is not None, f"Chosen child cannot be none for corresponding action root"
            corresponding_action_root = [c for c in current_node_root.children if
                                         c.action == chosen_child.action]
            if not len(corresponding_action_root) == 1:
                print(current_node_root.state)
                print(current_node_opponent.state)
            assert len(
                corresponding_action_root) == 1, f"Could not find corresponding root action {chosen_child.action}"
            chosen_child_root = corresponding_action_root[0]
        current_node_root = chosen_child_root  # Update Root Player Model node
        # State is chosen state after applying the action. It corresponds to the root player's original
        # private information as the (potentially wrong guessed) state was overwritten by the root player's
        # belief above
        root_player_state = chosen_child_root.state
        assert root_player_state is not None, "state should not be None"
        return current_node_root, root_player_state

    def _handle_choose_action_opponent(self, current_node_opponent, information_set_generator_opponent, root_opponent):
        state = current_node_opponent.state
        if not current_node_opponent.children:
            # For a new node, initialize its state, then choose a child as normal.
            legal_actions = self.evaluator.prior(state)
            if current_node_opponent is root_opponent and self._dirichlet_noise:
                epsilon, alpha = self._dirichlet_noise
                noise = self._random_state.dirichlet([alpha] * len(legal_actions))
                legal_actions = [(a, (1 - epsilon) * p + epsilon * n)
                                 for (a, p), n in zip(legal_actions, noise)]
            # Reduce bias from move generation order.
            self._random_state.shuffle(legal_actions)

            player = state.current_player()  # Player that has to choose an action
            for action, prior in legal_actions:
                state_copy = state.clone()  # Create clones of the state for every possible action
                if self.verbose:
                    print(state_copy.observation_string(0))
                state_copy.apply_action(action)
                if state_copy.is_terminal():
                    # Store state in observation node, Observation is useless as the game is already ended.
                    # This is necessary, as we do not model the resulting state after applying an action
                    # but just the resulting observation that this resulting state induces
                    # To be able to get the returns from terminal state, we include the state in observation
                    # nodes that result from terminal states
                    terminal_state = state_copy
                else:
                    terminal_state = None
                information_set_generator_copy = information_set_generator_opponent.clone()
                if self.verbose:
                    print(
                        f"info set: {[[state.observation_string(player_id) for state in set] for player_id, set in information_set_generator_copy.previous_information_set.items()]}")
                    print(information_set_generator_copy.action_history)
                    print(information_set_generator_copy._get_observation(state_copy, 1))
                information_set_generator_copy.register(state_copy, action)

                child_node = ImperfectInformationSearchNode.from_observation_tree_node(
                    information_set_generator_copy,
                    action,
                    player,  # Player that chose the action
                    prior,
                    terminal_state=terminal_state,
                    causing_state=state_copy,
                    deciding_player=state_copy.current_player())
                current_node_opponent.children.append(child_node)
            # This will be the last node to be considered in the Opponent Model Tree
        if state.is_chance_node():
            raise ValueError("State should never be chance node")
        else:
            # Otherwise choose node with largest UCT value
            chosen_child = max(
                current_node_opponent.children,
                key=lambda c: self._child_selection_fn(  # pylint: disable=g-long-lambda
                    c, current_node_opponent.explore_count, self.uct_c))
        if self.verbose:
            print(f"  - Player {current_node_opponent.player} chooses action {chosen_child.action}")
        # Set to chosen child node
        current_node_opponent = chosen_child  # next node will be observation node
        return chosen_child, current_node_opponent

    # -------------------------------------------------------------------------
    # Helper Methods: Chance Node handling
    # -------------------------------------------------------------------------

    @staticmethod
    def _handle_chance_node_guess(current_node):
        # Chance player handling
        if not current_node.children:
            current_node.children = [
                ImperfectInformationSearchNode.from_game_tree_node(
                    0,
                    -1,
                    # Player that chooses next action also guesses now from observation
                    0,
                    None  # .clone()
                )
            ]
        current_node = current_node.children[0]
        return current_node

    def _handle_chance_node(self, current_node, information_set_generator, player):
        if self.verbose:
            print(f'chance node select action player {player}')
        possible_states = dict()

        # Chance_player_actions should only contain the actions of the chance player that are still possible given the
        # private information of all players. This is fine as the MCTS search does not know the actual information set
        # of the opponent. Instead, the opponents information set is updated with regard to the root player's first
        # guess
        chance_player_actions = None
        for player_id in {0, 1, PUBLIC_OBSERVER_PLAYER_ID}:
            possible_states[player_id] = information_set_generator.calculate_information_set(player_id)
            all_action_masks = np.array([state.legal_actions_mask() for state in possible_states[player_id]])
            chance_player_actions_p = set(np.where(np.sum(all_action_masks, axis=0) > 0)[0])
            if chance_player_actions is None:
                chance_player_actions = chance_player_actions_p
            else:
                chance_player_actions = chance_player_actions.intersection(chance_player_actions_p)

        chance_player_actions = np.array(list(chance_player_actions))
        if not current_node.children:

            for action in chance_player_actions:
                information_set_generator_copy = information_set_generator.clone()
                information_set_generator_copy.register_chance_player_action(action)

                child_node = ImperfectInformationSearchNode.from_observation_tree_node(
                    information_set_generator_copy,
                    action,
                    -1,  # Player that chose the action
                    0,
                    terminal_state=None,
                    causing_state=None,
                    deciding_player=information_set_generator_copy.current_player())
                current_node.children.append(child_node)
        # For chance nodes, rollout according to chance node's probability
        # distribution
        # For every legal chance player action, loop through the information set states and add up the probabilities
        # of that chance player action
        all_chance_outcomes = np.zeros(self._game.max_chance_outcomes())
        for action in chance_player_actions:
            for player_id in {0, 1, PUBLIC_OBSERVER_PLAYER_ID}:
                for state in possible_states[player_id]:
                    actions, probs = zip(*state.chance_outcomes())
                    if action in actions:
                        all_chance_outcomes[action] += probs[actions.index(action)]

        all_chance_outcomes /= np.sum(all_chance_outcomes)
        assert np.sum(all_chance_outcomes) == 1.0, "Chance outcomes should sum to 1"
        legal_action_mask = np.zeros(len(possible_states[0][0].legal_actions_mask()), dtype=np.bool)
        legal_action_mask[chance_player_actions] = True
        assert np.sum(all_chance_outcomes[~legal_action_mask]) == 0, "Chance player about to select an illegal action"

        action = self._random_state.choice(range(self._game.max_chance_outcomes()),
                                           p=all_chance_outcomes)
        cc = next(
            c for c in current_node.children if c.action == action)
        current_node = cc
        return current_node

    def mcts_search(self, information_set_generator: InformationSetGenerator) -> ImperfectInformationSearchNode:
        """
        Extension of MCTS search for imperfect information games
        """
        root_player = information_set_generator.current_player()
        root = ImperfectInformationSearchNode.from_observation_tree_node(information_set_generator.clone(),
                                                                         None,
                                                                         None,  # We don't know who played the action
                                                                         1,
                                                                         deciding_player=root_player)
        root_root = ImperfectInformationSearchNode.from_observation_tree_node(information_set_generator.clone(),
                                                                              None,
                                                                              None,
                                                                              # We don't know who played the action
                                                                              1,
                                                                              deciding_player=root_player)
        for i in range(self.max_simulations):
            visit_path, leaf_node, visit_path_root, leaf_node_root = self._apply_tree_policy(root, root_root)

            returns = None
            solved = self.solve
            if leaf_node.is_terminal():
                returns = leaf_node.terminal_state.returns()
                visit_path[-1].outcome = returns
                solved = self.solve
            elif leaf_node.is_game_tree_node():
                returns = self.evaluator.evaluate(leaf_node.state)
                # Account for games with higher rewards
                returns = [returns[0] * self._game.max_utility(), returns[1] * self._game.max_utility()]
                solved = False
            elif leaf_node.is_observation_tree_node():
                returns = self.evaluator.evaluate_observation_node(leaf_node.information_set_generator)
                returns = [returns[0] * self._game.max_utility(), returns[1] * self._game.max_utility()]
                solved = False

            returns_root = None
            solved_root = self.solve
            if leaf_node_root.is_terminal():
                returns_root = leaf_node_root.terminal_state.returns()
                visit_path_root[-1].outcome = returns_root
                assert visit_path_root[-1].outcome is None or visit_path_root[-1].outcome == returns_root
                if visit_path[-1].actual_outcome:
                    visit_path[-1].actual_outcome.append(returns_root)
                else:
                    visit_path[-1].actual_outcome = [returns_root]
                solved_root = self.solve
            elif leaf_node_root.is_game_tree_node():
                returns_root = self.evaluator.evaluate(leaf_node_root.state)
                returns_root = [returns_root[0] * self._game.max_utility(), returns_root[1] * self._game.max_utility()]
                solved_root = False
            elif leaf_node_root.is_observation_tree_node():
                returns_root = self.evaluator.evaluate_observation_node(leaf_node_root.information_set_generator)
                returns_root = [returns_root[0] * self._game.max_utility(), returns_root[1] * self._game.max_utility()]
                solved_root = False

            root.player = root_player  # Technically, we don't know which player caused the root action
            root_root.player = root_player
            # Backup
            for node in reversed(visit_path):
                # node.total_reward += returns[root_player if node.player ==
                #                                             pyspiel.PlayerId.CHANCE else node.player]
                assert len(returns) == 2
                # TODO: Can I just use returns_root for the root player to make him choose more actions that are actually better for him? Given his private knowledge
                if node.player == root_player:
                    node.total_reward += returns_root[root_player]
                else:
                    node.total_reward += returns[root_player if node.player ==
                                                                pyspiel.PlayerId.CHANCE else node.player]

                node.actual_reward += returns_root[root_player if node.player ==
                                                                  pyspiel.PlayerId.CHANCE else node.player]
                node.explore_count += 1

                if solved and node.children:
                    # TODO: is this ever executed?
                    print('Backup')
                    player = node.children[0].player
                    # player = node.player  # TODO: I think, I changed it now such that the node's player is the player who is going to play the next action (not the one that played the action resulting in that node)
                    if player == pyspiel.PlayerId.CHANCE:
                        print('Chance node backup not tested!')
                        # TODO: actual rewards
                        # Only back up chance nodes if all have the same outcome.
                        # An alternative would be to back up the weighted average of
                        # outcomes if all children are solved, but that is less clear.
                        outcome = node.children[0].outcome
                        if (outcome is not None and
                                all(np.array_equal(c.outcome, outcome) for c in node.children)):
                            node.outcome = outcome
                        else:
                            solved = False
                    else:
                        # If any have max utility (won?), or all children are solved,
                        # choose the one best for the player choosing.
                        best = None
                        all_solved = True
                        for child in node.children:
                            if child.outcome is None:
                                all_solved = False
                            elif best is None or child.outcome[player] > best.outcome[player]:
                                best = child
                        if (best is not None and
                                (all_solved or best.outcome[player] == self.max_utility)):
                            node.outcome = best.outcome
                            node.actual_outcome = best.actual_outcome
                        else:
                            solved = False

            for node in reversed(visit_path_root):
                if node.player == 1 - root_player:
                    node.total_reward += returns[1 - root_player]
                else:
                    node.total_reward += returns_root[root_player if node.player ==
                                                                     pyspiel.PlayerId.CHANCE else node.player]
                node.explore_count += 1

                if solved_root and node.children:
                    player = node.children[0].player
                    # player = node.player  # TODO: I think, I changed it now such that the node's player is the player who is going to play the next action (not the one that played the action resulting in that node)
                    if player == pyspiel.PlayerId.CHANCE:
                        # Only back up chance nodes if all have the same outcome.
                        # An alternative would be to back up the weighted average of
                        # outcomes if all children are solved, but that is less clear.
                        outcome = node.children[0].outcome
                        if (outcome is not None and
                                all(np.array_equal(c.outcome, outcome) for c in node.children)):
                            node.outcome = outcome
                        else:
                            solved_root = False
                    else:
                        # If any have max utility (won?), or all children are solved,
                        # choose the one best for the player choosing.
                        best = None
                        all_solved = True
                        for child in node.children:
                            if child.outcome is None:
                                all_solved = False
                            elif best is None or child.outcome[player] > best.outcome[player]:
                                best = child
                        if (best is not None and
                                (all_solved or best.outcome[player] == self.max_utility)):
                            node.outcome = best.outcome
                        else:
                            solved_root = False
            if root.outcome is not None:
                break

        assert root.children[0].is_game_tree_node(), "First node should be game tree node"

        return root, root_root
