import torch.nn as nn
from rl_with_rnn import RNNTSP
from rl_mission_planning import HetNet, HetNet_tSNE, HetNet_Partial, HetNet_Partial_tSNE

from Environment.utils import get_solution_reward


class Solver(nn.Module):
    def __init__(self):
        super(Solver, self).__init__()

    def reward(self, sample_solution):
        """
        [Input]
            sample_solution: batch x seq_len x feature_size -> torch.LongTensor
        [Return]
            tour_len: batch
        """

        tour_len = get_solution_reward(sample_solution)

        return tour_len

    def forward(self, inputs):
        """
        [Input]
            inputs: batch x seq_len x feature
        [Return]
            R:
            probs:
            actions: batch x seq_len
        """

        probs, actions = self.actor(inputs)

        R = self.reward(inputs.gather(1, actions.unsqueeze(2).repeat(1, 1, inputs.shape[-1])))

        return R, probs, actions

class Solver_Partial(nn.Module):
    def __init__(self):
        super(Solver_Partial, self).__init__()

    def reward(self, sample_solution):
        """
        [Input]
            sample_solution: batch x seq_len x feature_size -> torch.LongTensor
        [Return]
            tour_len: batch
        """

        tour_len = get_solution_reward(sample_solution)

        return tour_len

    def forward(self, inputs):
        """
        [Input]
            inputs: batch x seq_len x feature
        [Return]
            R:
            probs:
            actions: batch x seq_len
        """

        probs, actions = self.actor(inputs)

        R = self.reward(inputs.gather(1, actions.unsqueeze(2).repeat(1, 1, inputs.shape[-1])))

        return R, probs, actions


class Solver_tSNE(nn.Module):
    def __init__(self):
        super(Solver_tSNE, self).__init__()

    def reward(self, sample_solution):
        """
        [Input]
            sample_solution: batch x seq_len x feature_size -> torch.LongTensor
        [Return]
            tour_len: batch
        """

        tour_len = get_solution_reward(sample_solution)

        return tour_len

    def forward(self, inputs):
        """
        [Input]
            inputs: batch x seq_len x feature
        [Return]
            R:
            probs:
            actions: batch x seq_len
        """

        probs, actions, embeddings = self.actor(inputs)

        R = self.reward(inputs.gather(1, actions.unsqueeze(2).repeat(1, 1, inputs.shape[-1])))

        return R, probs, actions, embeddings


class Solver_Partial_tSNE(nn.Module):
    def __init__(self):
        super(Solver_Partial_tSNE, self).__init__()

    def reward(self, sample_solution):
        """
        [Input]
            sample_solution: batch x seq_len x feature_size -> torch.LongTensor
        [Return]
            tour_len: batch
        """

        tour_len = get_solution_reward(sample_solution)

        return tour_len

    def forward(self, inputs):
        """
        [Input]
            inputs: batch x seq_len x feature
        [Return]
            R:
            probs:
            actions: batch x seq_len
        """

        probs, actions, embeddings = self.actor(inputs)

        R = self.reward(inputs.gather(1, actions.unsqueeze(2).repeat(1, 1, inputs.shape[-1])))

        return R, probs, actions, embeddings


class solver_RNN(Solver):
    def __init__(self,
            embedding_size,
            hidden_size,
            seq_len,
            n_glimpses,
            tanh_exploration):
        super(solver_RNN, self).__init__()

        self.actor = RNNTSP(embedding_size,
                                hidden_size,
                                seq_len,
                                n_glimpses,
                                tanh_exploration)


class HetNet_solver(Solver):
    def __init__(self,
                embedding_size,
                hidden_size,
                n_glimpses,
                tanh_exploration):
            super(HetNet_solver, self).__init__()

            self.actor = HetNet(embedding_size,
                                    hidden_size)

class HetNet_Partial_solver(Solver_Partial):
    def __init__(self,
                partial_type,
                input_size,
                embedding_size,
                hidden_size,
                n_glimpses,
                tanh_exploration):
            super(HetNet_Partial_solver, self).__init__()

            self.actor = HetNet_Partial(partial_type,
                                        input_size,
                                        embedding_size,
                                        hidden_size)


class HetNet_tSNE_solver(Solver_tSNE):
    def __init__(self,
                embedding_size,
                hidden_size,
                n_glimpses,
                tanh_exploration):
            super(HetNet_tSNE_solver, self).__init__()

            self.actor = HetNet_tSNE(embedding_size,
                                    hidden_size)


class HetNet_Partial_tSNE_solver(Solver_Partial_tSNE):
    def __init__(self,
                partial_type,
                input_size,
                embedding_size,
                hidden_size,
                n_glimpses,
                tanh_exploration):
            super(HetNet_Partial_tSNE_solver, self).__init__()

            self.actor = HetNet_Partial_tSNE(partial_type,
                                             input_size,
                                             embedding_size,
                                             hidden_size)
