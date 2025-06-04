from typing import Optional, List, Tuple
from uuid import UUID

class TreeNode:
    def __init__(self, point: List[float], chunk_id: UUID, left=None, right=None, axis=0):
        self.point = point
        self.chunk_id = chunk_id
        self.left = left
        self.right = right
        self.axis = axis



    