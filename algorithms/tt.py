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