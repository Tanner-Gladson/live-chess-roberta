
import chess
import random
import torch
from typing import Any, Optional, Union
import logging
import pandas as pd
import sys
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForMaskedLM,
    PreTrainedTokenizerFast,
    AutoTokenizer,
    pipeline,
    BatchEncoding,
)

logger = logging.getLogger(__name__)


class ChessRobertaDriver:
  '''A wrapper for the ChessRoberta huggingface model'''
  MAX_NUM_PADDING_FRAMES_TO_LOAD = 1000 # The frames which will be randomly used to fill out the context window
  MAX_SANS_FOR_INFERENCE_GAME = 7 # The number of half-moves to be kept in the target_frame during inference
  BATCH_SIZE = 16 # The number of possible moves to sameple from the model

  def __init__(self):
    print("Initializing a ChessRobertaDriver instance")
    # TODO: we need a small corpus to sample from for packing sequences
    dataset = load_dataset("TannerGladson/lichess-frames", split="train", streaming=True)
    self.sample_frames : list[str] = self.get_samples(dataset)
    print(self.sample_frames)
    raise NotImplementedError("stop here")

    # initialize ChessRoberta
    if torch.cuda.is_available():
      print("using cuda")
      device = torch.device("cuda")
    else:
      print("using cpu")
      device = torch.device("cpu")
  
    self.tokenizer = AutoTokenizer.from_pretrained("TannerGladson/chess-roberta")
    self.model = AutoModelForMaskedLM.from_pretrained("TannerGladson/chess-roberta")
    self.model.to(device)
    print("ChessRobertaDriver parameters have been loaded")


  def get_next_play(self, moves: list[str], board: chess.Board) -> chess.Move:
    '''Return the next UCI move in a game. Assumes moves contains all moves from starting FEN
    Parameters:
      - moves: a list of UCI moves from the starting FEN
    '''
    generated_moves : list[chess.Move] = self.generate_move_candidates()
    all_legal_moves = board.legal_moves
    legal_generated_moves = [move for move in generated_moves if move in all_legal_moves]

    if len(legal_generated_moves) == 0:
      logger.info("ChessRobertaDriver did not generate any legal moves, choosing random move")
      move : chess.Move = random.choice(all_legal_moves)
    else:
      move : chess.Move = random.choice(legal_generated_moves)
    return moves
  
  def generate_move_candidates(self, moves: list[str]) -> list[chess.Move]:
    ''' Generate a batch of candidates for the next move'''
    # TODO: from this game's list of moves, get the PGN of the last min(len(moves), MAX_SANS_FOR_INFERENCE_GAME) half-moves
    # TODO ... also get the corresponding FEN

    # TODO: construct the lichess frame from these PGNs and FENs
    target_frame : str = ""

    # TODO: generated a batch of 16 tokenized sequences, where target_frame

    # TODO: sample from the ChessRoberta transformer to get the next 7 predicted tokens

    # TODO: parse those tokens to identify at most the next move

    # TODO: convert this move from PGN to UCI using the chess library
    
    return []

  def generate_tokenized_batch(self, target_frame: str) -> BatchEncoding:
    '''Generate BATCH_SIZE tokenized sequences where target_frame is the first frame in each sample'''

    # TODO: can copy from the tokenize_and_batch_frames() function in testing_inference.py

    return BatchEncoding(result)
  
  @staticmethod
  def get_samples(streaming_dataset) -> list[str]:
    '''Get up to N random samples from a dataset'''
    CHANCE_TO_KEEP = 0.001 # 0.1%
    sample_frames = []
    
    for _, item in enumerate(streaming_dataset):
      if len(sample_frames) == ChessRobertaDriver.MAX_NUM_PADDING_FRAMES_TO_LOAD:
        break
      if random.random() < CHANCE_TO_KEEP:
        sample_frames.append(item["text"])

    return sample_frames