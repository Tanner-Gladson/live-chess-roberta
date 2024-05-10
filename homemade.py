"""
Some example classes for people who want to create a homemade bot.

With these classes, bot makers will not have to implement the UCI or XBoard interfaces themselves.
"""

from __future__ import annotations
import chess
from chess.engine import PlayResult, Limit
import random
from lib.engine_wrapper import MinimalEngine, MOVE
from typing import Any, Optional, Union
import logging
import sys
from lib.config import Configuration
from lib import model

OPTIONS_TYPE = dict[str, Any]
COMMANDS_TYPE = list[str]
MOVE = Union[chess.engine.PlayResult, list[chess.Move]]
logger = logging.getLogger(__name__)

class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""
    pass

class ChessRobertaEngine(ExampleEngine):
    """Chess Roberta! But for now, get a random move."""

    def __init__(self, commands: COMMANDS_TYPE, options: OPTIONS_TYPE, stderr: Optional[int],
                draw_or_resign: Configuration, game: Optional[model.Game], **popen_args: str):
        """Start Stockfish."""
        super().__init__(commands, options, stderr, draw_or_resign, game, **popen_args)
        
        self.chess_roberta_driver = None
        
    # def search(self, board: chess.Board, game: model.Game, time_limit: chess.engine.Limit, ponder: bool, draw_offered: bool,
    #         root_moves: MOVE) -> chess.engine.PlayResult:
    #     """Get a move using Stockfish."""
    #     return self.engine.play(board, time_limit)

    def search(self, board: chess.Board, *args: Any) -> PlayResult:
        """Choose a random move."""
        return PlayResult(random.choice(list(board.legal_moves)), None)