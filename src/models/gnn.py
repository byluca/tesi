"""
Graph Neural Networks implementation adapted from torch_geometric.nn.
"""
import torch
from typing import Tuple, Union
from torch_geometric.nn.conv import ChebConv, GATConv, SAGEConv


__all__ = ['GraphSAGE', 'GAT', 'ChebNet']


class GraphSAGE(torch.nn.Module):
	def __init__(
			self,
			in_channels: Union[int, Tuple[int, int]],
			hidden_channels: Tuple[int],
			out_channels: int,
			dropout: float,
			aggr: str,
			name: str = 'GraphSAGE'
		):
		super(GraphSAGE, self).__init__()
		self.layers = torch.nn.ModuleList()
		self.dropout = torch.nn.Dropout(p=dropout)
		self.name = name

		# Input layer
		self.layers.append(SAGEConv(in_channels, hidden_channels[0], aggr=aggr))

		# Hidden layers
		for i in range(len(hidden_channels) - 1):
			self.layers.append(SAGEConv(hidden_channels[i], hidden_channels[i+1], aggr=aggr))

		# Output layer
		self.layers.append(SAGEConv(hidden_channels[-1], out_channels, aggr=aggr))

	def forward(self, x, edge_index):
		for conv in self.layers[:-1]:
			x = conv(x, edge_index).relu()
			x = self.dropout(x)
		x = self.layers[-1](x, edge_index)
		return x


class GAT(torch.nn.Module):
	def __init__(
			self,
			in_channels: Union[int, Tuple[int, int]],
			hidden_channels: Tuple[int],
			out_channels: int,
			dropout: float,
			heads: int,
			attention_dropout: float,
			name: str = 'GAT'
		):
		super(GAT, self).__init__()
		self.layers = torch.nn.ModuleList()
		self.dropout = torch.nn.Dropout(p=dropout)
		self.name = name

		# Input layer
		self.layers.append(GATConv(in_channels, hidden_channels[0], heads=heads, dropout=attention_dropout))

		# Hidden layers
		for i in range(len(hidden_channels) - 1):
			self.layers.append(GATConv(hidden_channels[i] * heads, hidden_channels[i+1], heads=heads, dropout=attention_dropout))

		# Output layer
		self.layers.append(GATConv(hidden_channels[-1] * heads, out_channels, heads=1, concat=False, dropout=attention_dropout))

	def forward(self, x, edge_index):
		for conv in self.layers[:-1]:
			x = conv(x, edge_index).relu()
			x = self.dropout(x)
		x = self.layers[-1](x, edge_index)
		return x


class ChebNet(torch.nn.Module):
	def __init__(
			self,
			in_channels: Union[int, Tuple[int, int]],
			hidden_channels: Tuple[int],
			out_channels: int,
			dropout: float,
			K: int,
			name: str = 'ChebNet'
		):
		super(ChebNet, self).__init__()
		self.layers = torch.nn.ModuleList()
		self.dropout = torch.nn.Dropout(p=dropout)
		self.name = name

		# Input layer
		self.layers.append(ChebConv(in_channels, hidden_channels[0], K))

		# Hidden layers
		for i in range(len(hidden_channels) - 1):
			self.layers.append(ChebConv(hidden_channels[i], hidden_channels[i+1], K))

		# Output layer
		self.layers.append(ChebConv(hidden_channels[-1], out_channels, K))

	def forward(self, x, edge_index):
		for conv in self.layers[:-1]:
			x = conv(x, edge_index).relu()
			x = self.dropout(x)
		x = self.layers[-1](x, edge_index)
		return x
