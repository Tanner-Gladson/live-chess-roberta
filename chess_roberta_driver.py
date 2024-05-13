
import chess
import random
from typing import Any, Optional, Union
import logging
import sys
logger = logging.getLogger(__name__)


class ChessRobertaDriver:
  '''A wrapper for the ChessRoberta huggingface model'''
  def __init__(self):
    print("Initializing a ChessRobertaDriver instance")
    # TODO: initialize the ChessRoberta huggingface pytorch model

  def get_next_play(moves: list[str], board: chess.Board) -> chess.Move:
    '''Return the next UCI move in a game. Assumes moves contains all moves from starting FEN
    Parameters:
      - moves: a list of UCI moves from the starting FEN
    '''
    # TODO: from this game's list of moves, get the PGN of the last min(4, len(moves)) half-moves

    # TODO ... also get the corresponding FEN

    # TODO: construct the lichess frame from these PGNs and FENs

    # TODO: sample from the ChessRoberta transformer to get the next 7 predicted tokens
    # -> do I need to call the tokenizer on it?
    # -> where do I apply the masking?

    # TODO: parse those tokens to identify at most the next move

    # TODO: convert this move from PGN to UCI using the chess library

    move : chess.Move = random.choice(list(board.legal_moves))
    return moves


    @staticmethod
    def get_legal_moves(board: chess.Board) -> list[chess.Move]:
        return list(board.legal_moves)