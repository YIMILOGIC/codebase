class Solution:
    # LC 68
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        res = []
        n = len(words)
        i = 0
        while i < n:
            begin = i
            total_chars = 0
            while i < n and total_chars + len(words[i]) + (i-begin) <= maxWidth:
                total_chars += len(words[i])
                i += 1

            if i == n or i - begin == 1:
                # only one word or last line
                line = " ".join(words[begin:i])
                n_suffix_space = maxWidth - len(line)
                if n_suffix_space != 0:
                    line += " " * n_suffix_space
                res.append(line)
            else:
                n_space = (maxWidth - total_chars) // (i - begin - 1)
                n_extra = (maxWidth - total_chars) % (i - begin - 1)

                line = words[begin]
                for j in range(begin+1, i):
                    extra = 1 if j-begin <= n_extra else 0
                    line += " " * (n_space + extra) + words[j]
                res.append(line)
        return res

    # LC 2021
    def brightestPosition(self, lights: List[List[int]]) -> int:
        # sweep line
        pos_light = defaultdict(int)
        for p, r in lights:
            pos_light[p-r] += 1
            pos_light[p+r+1] -= 1
        cur, max_pos, max_val = 0, -1, float("-inf")
        for pos, val in sorted(pos_light.items()):
            cur += val
            if cur > max_val:
                max_val, max_pos = cur, pos
        return max_pos

    # LC 162
    def findPeakElement(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] > nums[mid+1]:
                right = mid
            else:
                left = mid + 1
        return left

    # LC 1901
    def findPeakGrid(self, mat: List[List[int]]) -> List[int]:
        def get_max(nums):
            max_val = float("-inf")
            max_idx = 0
            for i in range(len(nums)):
                if nums[i] > max_val:
                    max_val = nums[i]
                    max_idx = i
            return max_idx

        top, bottom = 0, len(mat) - 1;
        while top < bottom:
            mid = top + (bottom - top) // 2
            max_idx = get_max(mat[mid])
            if mat[mid][max_idx] > mat[mid+1][max_idx]:
                bottom = mid
            else:
                top = mid + 1
             
        return [top, get_max(mat[top])]

    # LC 1091
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        if grid is None or len(grid) == 0 or len(grid[0]) == 0:
            return -1
        if grid[0][0] != 0 or grid[-1][-1] != 0:
            return -1
        m = len(grid)
        n = len(grid[0])
        queue = deque()
        queue.append((0,0))
        grid[0][0] = 1
        dirs = [(1,0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        res = 1
        while queue:
            r, c = queue.popleft()
            dist = grid[r][c]
            if r == m - 1 and c == n - 1:
                return dist
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == 0:
                    queue.append((nr, nc))
                    grid[nr][nc] = dist + 1
                    if nr == m - 1 and nc == n - 1:
                        return grid[nr][nc]

        return -1
    
    # LC 739
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        res = [0] * n
        stack = []
        for i in range(n):
            temp = temperatures[i] # save the index of temperature that has not found a warmer one.
            while stack and temp > temperatures[stack[-1]]:
                prev_idx = stack.pop()
                res[prev_idx] = i - prev_idx
            stack.append(i)
        return res
    
    # LC 496
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        d = {nums1[i]:i for i in range(len(nums1))}
        res = [-1] * len(nums1)
        stack = []
        for num in nums2:
            while stack and num > stack[-1]:
                cur = stack.pop()
                if cur in d:
                    res[d[cur]] = num
            stack.append(num)
        return res

    # LC 1827
    def minOperations(self, nums: List[int]) -> int:
        if not nums:
            return -1
        res = 0
        n = len(nums)
        for i in range(1, n):
            diff = nums[i] - nums[i-1]
            if nums[i] - nums[i-1] <= 0:
                nums[i] = nums[i-1] + 1
                res += (-diff + 1)
        return res

    # LC 238. Product of Array Except Self
    # TC: O(N) SC: O(N)
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = [0] * n
        left = [0] * n
        right = [0] * n

        left[0] = right[-1] = 1
        for i in range(1, n):
            left[i] = left[i-1] * nums[i-1]
        for i in range(n-2, -1, -1):
            right[i] = right[i+1] * nums[i+1]
        for i in range(n):
            res[i] = left[i] * right[i]
        return res

    # LC 36. Valid Sudoku
    # TC: O(N**2) SC: O(N)
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        if not board or not board[0]:
            return False
        m, n = len(board), len(board[0])
        if m != n:
            return False
        
        for i in range(m):
            visited = set()
            for j in range(m):
                if board[i][j] != ".":
                    val = int(board[i][j])
                    if val not in visited:
                        visited.add(val)
                    else:
                        return False
        
        for i in range(n):
            visited = set()
            for j in range(m):
                if board[j][i] != ".":
                    val = int(board[j][i])
                    if val not in visited:
                        visited.add(val)
                    else:
                        return False
        
        for i in range(0, m, 3):
            for j in range(0, n, 3):
                visited = set()
                for x in range(3):
                    for y in range(3):
                        if board[i+x][j+y] != '.':
                            val = int(board[i+x][j+y])
                            if val not in visited:
                                visited.add(val)
                            else:
                                return False
        return True

    # LC 236. Lowest Common Ancestor of a Binary Tree
    # TC: O(N) SC: O(H) N is number of nodes in tree and H is the height of the tree
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        def lca_helper(root, p, q):
            if not root or root == p or root == q:
                return root
            left = lca_helper(root.left, p, q)
            right = lca_helper(root.right, p, q)
            if left and right:
                return root
            if left:
                return left
            if right:
                return right
            return None
        
        if not root:
            return root
        lca = lca_helper(root, p, q)
        if lca == p:
            if lca_helper(p, q, q) == q:
                return p
        elif lca == q:
            if lca_helper(q, p, p) == p:
                return q
        return lca
