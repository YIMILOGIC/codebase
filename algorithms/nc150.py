class Solution:
    # LC 217. Contains Duplicate
    # TC: O(N) SC: O(N)
    # hashset
    def containsDuplicate(self, nums: List[int]) -> bool:
        if not nums:
            return False
        visited = set()
        for num in nums:
            if num in visited:
                return True
            else:
                visited.add(num)
        return False
    # LC 242. Valid Anagram
    # TC: O(N) SC: O(N)
    # hashset
    def isAnagram(self, s: str, t: str) -> bool:
        cnt_s = defaultdict(int)
        cnt_t = defaultdict(int)
        for c in s:
            cnt_s[c] += 1
        for c in t:
            cnt_t[c] += 1
        return cnt_s == cnt_t

    # LC 1. Two Sum
    # TC: O(N) SC: O(N)
    # use hashset
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        res = []
        visited = {}
        for i, num in enumerate(nums):
            if target - num not in visited:
                visited[num] = i
            else:
                res = [visited[target - num], i]
                break
        return res       

    # LC 49. Group Anagrams
    # TC: O(NKlogK) SC: O(NK) N: number of elements in strs K: max length of the string in strs
    # push (count, num) into a priority queue (min heap)
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        res = []
        if not strs:
            return res
        groups = {}
        for s in strs:
            cnt = [0] * 26
            for c in s:
                cnt[ord(c) - ord('a')] += 1
            cur_cnt = tuple(cnt)
            if cur_cnt not in groups:
                groups[cur_cnt] = [s]
            else:
                groups[cur_cnt].append(s)
        for cnt in groups:
            res.append(groups[cnt])
        return res 

    # LC 347. Top K Frequent Elements
    # TC: O(Nlogk) SC: O(N)
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        if not nums:
            return []
        pq = []
        counter = Counter(nums)
        for num, cnt in counter.items():
            if len(pq) < k:
                heapq.heappush(pq, (cnt, num))
            else:
                if cnt > pq[0][0]:
                    heapq.heapreplace(pq, (cnt, num))
        res = [0] * len(pq)
        for i in range(len(pq)-1, -1, -1):
            cur = heapq.heappop(pq)
            res[i] = cur[-1]
        return res

    # LC 271. Encode and Decode Strings
    # TC: O(N) SC: O(N+k) N is the sum of number of characters in strs, k is the the size of strs
    # add length information
    def encode(self, strs: List[str]) -> str:
        """Encodes a list of strings to a single string.
        """
        res = ''
        for s in strs:
            res += str(len(s)) + '#' + s
        return res
        

    def decode(self, s: str) -> List[str]:
        """Decodes a single string to a list of strings.
        """
        res = []
        i = 0
        j = 0
        while j < len(s):
            while s[j] != '#':
                j += 1
            # now s[j] == '#'
            length = int(s[i:j])
            j += 1
            res.append(s[j:j+length])
            i = j + length
            j = i
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

    # LC 128. Longest Consecutive Sequence
    # TC: O(N) SC: O(N)
    # use hashset, if a num-1 in hashset, then [num, num+k] will be count when we count from num-1.
    # there is no need to make duplicate count.
    def longestConsecutive(self, nums: List[int]) -> int:
        num_set = set(nums)
        res = 0
        for num in num_set:
            if num - 1 not in num_set:    
                cur_len = 1
                cur_num = num
                while cur_num + 1 in num_set:
                    cur_len += 1
                    cur_num += 1
                res = max(cur_len, res)
        return res

    # LC 125. Valid Palindrome 
    # TC: O(N) SC: O(1)
    # two pointers but pay attention to lowercase/uppercase
    def isPalindrome(self, s: str) -> bool:
        if s is None:
            return False
        i, j = 0, len(s) - 1
        while i < j:
            while i < j and not s[i].isalnum():
                i += 1
            while i < j and not s[j].isalnum():
                j -= 1
            if s[i].lower() != s[j].lower():
                return False
            i += 1
            j -= 1
        return True

    # LC 167. Two Sum II - Input Array Is Sorted
    # TC: O(N) SC: O(1)
    # two pointers
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        i, j = 0, len(numbers) - 1
        while i < j:
            cur_sum = numbers[i] + numbers[j]
            if cur_sum < target:
                i += 1
            elif cur_sum > target:
                j -= 1
            else:
                return [i + 1, j + 1]

    # LC 167. Two Sum II - Input Array Is Sorted
    # TC: O(N) SC: O(1)
    # two pointers
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        i, j = 0, len(numbers) - 1
        while i < j:
            cur_sum = numbers[i] + numbers[j]
            if cur_sum < target:
                i += 1
            elif cur_sum > target:
                j -= 1
            else:
                return [i + 1, j + 1]

    # LC 15. 3Sum
    # TC: O(N**2) SC: O(N)
    # sort the array, for each unique element in nums, use two sum II to find all triplets
    # if a nums[i] == nums[i-1] then ignore nums[i]
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        def two_sum(nums: List[int], index: int, res: List[List[int]]):
            # for sorted array/list
            j, k = index+1, len(nums) - 1
            while j < k:
                cur_sum = nums[j] + nums[k] + nums[index]
                if cur_sum > 0:
                    k -= 1
                elif cur_sum < 0:
                    j += 1
                else:
                    res.append([nums[index], nums[j], nums[k]])
                    while j+1 < len(nums) and nums[j+1] == nums[j]:
                        j += 1
                    j += 1
                    k -= 1

        res = []
        nums.sort()
        for i in range(len(nums)):
            if nums[i] > 0:
                break
            if i == 0 or nums[i-1] != nums[i]:
                two_sum(nums, i, res)
        return res
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res, dups = set(), set()
        seen = {}
        for i, val1 in enumerate(nums):
            if val1 not in dups:
                dups.add(val1)
                for j, val2 in enumerate(nums[i + 1:]):
                    complement = -val1 - val2
                    if complement in seen and seen[complement] == i:
                        res.add(tuple(sorted((val1, val2, complement))))
                    seen[val2] = i
        return [list(x) for x in res]

    # LC 11. Container With Most Water
    # TC: O(N) SC: O(N)
    # always change the shorter bar
    def maxArea(self, height: List[int]) -> int:
        i, j = 0, len(height)-1
        res = 0
        while i < j:
            area = min(height[i], height[j]) * (j-i)
            res = max(area, res)
            if height[i] <= height[j]:
                i += 1
            else:
                j -= 1
        return res

    # LC 543. Diameter of Binary Tree
    # TC: O(N) SC: O(H)
    # what would a node get from a child: the height/depth of the subtree (child is the root)
    # the operation at current node: compute the number of nodes on longest diameter where current node is the root of subtree
    # return value: the max depth (max(left, right)+1) of current subtree
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        
        def helper(root, res):
            if not root:
                return 0
            left = helper(root.left, res)
            right = helper(root.right, res)
            res[0] = max(res[0], left + right + 1)
            return max(left, right) + 1
        
        res = [0]
        helper(root, res)
        return res[0] - 1

    # LC 71. Simplify Path
    # TC: O(N) SC: O(N)

    def simplifyPath(self, path: str) -> str:
        stack = []
        parts = path.split("/")
        for s in parts:
            if s == '..':
                if stack:
                    stack.pop()
            elif s == '.':
                continue
            elif s != '':
                stack.append(s)
        res = ""
        for s in stack:
            res += "/" + s
        return res if len(res) else "/"


    # LC 3. Longest Substring Without Repeating Characters
    # TC: O(N) SC: O(N)
    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s:
            return 0
        char_pos = {}
        i = 0
        j = 0
        res = 0
        while j < len(s):
            if s[j] in char_pos:
                i = max(char_pos[s[j]]+1, i)
            res = max(res, j - i + 1)

            char_pos[s[j]] = j
            j += 1
        return res

    # LC 88. Merge Sorted Array
    # TC: O(m+n) SC: O(1)
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        k = m + n - 1
        j = n - 1
        i = m - 1
        while i >= 0 and j >= 0:
            if nums1[i] >= nums2[j]:
                nums1[k] = nums1[i]
                i -= 1
            else:
                nums1[k] = nums2[j]
                j -= 1
            k -= 1
        while j >= 0:
            nums1[j] = nums2[j]
            j -= 1
            
    # LC 424. Longest Repeating Character Replacement
    # TC: O(N) SC: O(1)
    # sliding window, two pointers, the length of window is never reduced.
    # no need to update cur_max after i moved: if there is a longer results, there must be a larger cur_max
    def characterReplacement(self, s: str, k: int) -> int:
        i, j = 0, 0
        ch_freq = [0] * 26
        cur_max = 0
        res = 0

        while j < len(s):
            idx = ord(s[j]) - ord('A')
            ch_freq[idx] += 1
            cur_max = max(cur_max, ch_freq[idx])
            if cur_max + k < j - i + 1:
                idx = ord(s[i]) - ord('A')
                ch_freq[idx] -= 1
                i += 1
            res = max(res, j - i + 1)
            j += 1
        return res

    # LC 88. Merge Sorted Array
    # TC: O(N) SC: O(1)
    # merge two arrays from the end so no extra space is needed
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        k = m + n - 1
        j = n - 1
        i = m - 1
        while i >= 0 and j >= 0:
            if nums1[i] >= nums2[j]:
                nums1[k] = nums1[i]
                i -= 1
            else:
                nums1[k] = nums2[j]
                j -= 1
            k -= 1
        while j >= 0:
            nums1[j] = nums2[j]
            j -= 1
            
    # LC 567. Permutation in String
    # TC: O(len(s2)) SC: O(26) = O(1)
    # also can use a array of size 26 for the hashmap
    def checkInclusion(self, s1: str, s2: str) -> bool:
        n1 = len(s1)
        n2 = len(s2)
        if n2 < n1:
            return False
        c1 = Counter(s1)
        c2 = Counter(s2[:n1])
        if c1 == c2:
            return True
        for i in range(n1, n2):
            c2[s2[i-n1]] -= 1
            c2[s2[i]] += 1
            if c1 == c2:
                return True
        return False

    # LC 875. Koko Eating Bananas
    # TC: O(NlogM) SC: O(1) N is the size of input array piles, m is the largest number of bananas in a single pile in piles.
    # Binary search
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        def hours_to_finish(piles, k):
            hours = 0
            for p in piles:
                cur_hour = p // k if p % k == 0 else p // k + 1
                hours += cur_hour
            return hours


        left = 1 
        right = max(piles)
        while left < right:
            mid = left + (right - left) // 2
            hours = hours_to_finish(piles, mid)
            if hours <= h:
                right = mid
            else:
                left = mid + 1
        return left

    # LC 239. Sliding Window Maximum
    # TC: O(N), SC: O(k)
    # monotonic deque
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if not nums or k == 0:
            return []
        dq = deque()
        res = []
        # initialize deque
        for i in range(k):
            while dq and dq[-1] < nums[i]:
                dq.pop()
            dq.append(nums[i])
        res.append(dq[0])
        # process the rest
        for i in range(k, len(nums)):
            if dq[0] == nums[i - k]:
                dq.popleft()
            while dq and dq[-1] < nums[i]:
                dq.pop()
            dq.append(nums[i])
            res.append(dq[0])
        return res

    # LC 162. Find Peak Element
    # TC: O(logN), SC: O(1)
    # compare nums[mid] with nums[mid+1]
    def findPeakElement(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] > nums[mid + 1]:
                right = mid
            else:
                left = mid + 1
        return left

    # LC 680. Valid Palindrome II
    # TC: O(N), SC: O(1). check_valid() is called only twice
    def validPalindrome(self, s: str) -> bool:
        def check_valid(s, i, j):
            while i < j:
                if s[i] != s[j]:
                    return False
                i += 1
                j -= 1
            return True
        
        i, j = 0, len(s) - 1
        while i < j:
            if s[i] != s[j]:
                return check_valid(s, i, j-1) or check_valid(s, i+1, j)
            i += 1
            j -= 1
        return True

    # LC 215. Kth Largest Element in an Array
    # TC: O(Nlogk) SC:O(k)
    def findKthLargestMinHeap(self, nums: List[int], k: int) -> int:
        pq = []
        for num in nums:
            heapq.heappush(pq, num)
            if len(pq) > k:
                heapq.heappop(pq)
        return pq[0]

    # LC 215. Kth Largest Element in an Array
    # TC: O(N), SC: O(1)
    # quick select
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def quick_select(nums, left, right, k):
            pivot = nums[right]
            l = left
            r = right
            i = left
            while i <= r:
                if nums[i] < pivot:
                    nums[l], nums[i] = nums[i], nums[l]
                    l += 1
                    i += 1
                elif nums[i] > pivot:
                    nums[i], nums[r] = nums[r], nums[i]
                    r -= 1
                else:
                    i += 1
            n_small = l - left
            n_large = right - r
            n_equal = r - l + 1
            if k <= n_large:
                return quick_select(nums, r + 1, right, k)
            elif k > n_large + n_equal:
                return quick_select(nums, left, l - 1, k - n_large - n_equal)
            else:
                return pivot

        return quick_select(nums, 0, len(nums) - 1, k)

    # LC 236. Lowest Common Ancestor of a Binary Tree
    # TC: O(N), SC: O(H), where H is the height of the tree
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, x):
    #         self.val = x
    #         self.left = None
    #         self.right = None
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        def lca_helper(root, p, q):
            if not root or root == p or root == q:
                return root
            left = lca_helper(root.left, p, q)
            right = lca_helper(root.right, p, q)
            if left and right:
                return root
            elif left:
                return left
            elif right:
                return right
            else:
                return None

        if not root:
            return None
        lca = lca_helper(root, p, q)
        if lca == p:
            if lca_helper(root, q, q) == q:
                return p
        elif lca == q:
            if lca_helper(root, p, p) == p:
                return q
        return lca

    # LC 1650. Lowest Common Ancestor of a Binary Tree III
    # TC: O(H), SC: O(1)
    # compute height, go along the lower node to the same height of the higher node, then follow parent and compare
    def lowestCommonAncestorIII(self, p: 'Node', q: 'Node') -> 'Node':
        def height(node):
            h = 0 
            cur = node
            while cur:
                h += 1
                cur = cur.parent
            return h
        hp = height(p)
        hq = height(q)
        print(hp, hq)
        low = p
        high = q
        if hp < hq:
            low = q
            high = p
        diff = abs(hp - hq)
        
        while diff > 0 and low:
            low = low.parent
            diff -= 1
        print(low.val, high.val)
        while low and high:
            if low == high:
                return low
            low = low.parent
            high = high.parent
        return None

    # LC 314. Binary Tree Vertical Order Traversal
    # TC: O(N), SC:O(N)
    # BFS and save column number
    def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        visited = defaultdict(list)
        q = deque()
        if root:
            q.append((root, 0))
        min_col = float("inf")
        max_col = float("-inf")
        while q:
            node, col = q.popleft()
            visited[col].append(node.val)
            min_col = min(min_col, col)
            max_col = max(max_col, col)
            if node.left:
                q.append((node.left, col - 1))
            if node.right:
                q.append((node.right, col + 1))
        res = []
        if len(visited) != 0:
            for i in range(min_col, max_col + 1):
                if i in visited:
                    res.append(visited[i])
        return res

    # LC 1249. Minimum Remove to Make Valid Parentheses
    # TC: O(N), SC: O(N)
    # two pass: remove invalid symbol forward and backward
    def minRemoveToMakeValid(self, s: str) -> str:
        def delete_invalid(s: str, open_symbol: str, close_symbol):
            res = []
            balance = 0
            for c in s:
                if c == open_symbol:
                    balance += 1
                if c == close_symbol:
                    if balance == 0:
                        continue
                    balance -= 1
                res.append(c)
            return "".join(res)
        res = delete_invalid(s, "(", ")")
        res = delete_invalid(res[::-1], ")", "(")
        return res[::-1]

    # LC 938. Range Sum of BST
    # TC: O(N), SC:O(H)
    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        def helper(root, low, high, res):
            if not root:
                return
            if root.val > high:
                helper(root.left, low, high, res)
            elif root.val < low:
                helper(root.right, low, high, res)
            else:
                res[0] += root.val
                helper(root.left, low, high, res)
                helper(root.right, low, high, res)
        
        res = [0]
        helper(root, low, high, res)
        return res[0]

    # LC 973. K Closest Points to Origin
    # TC: O(N), SC: O(k)
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        pq = []
        for p in points:
            dist = p[0] ** 2 + p[1] ** 2
            heapq.heappush(pq, (-dist, p))
            if len(pq) > k:
                heapq.heappop(pq)
        res = [x[1] for x in pq]
        return res

    # LC 408. Valid Word Abbreviation
    # TC: O(N), O(1)
    def validWordAbbreviation(self, word: str, abbr: str) -> bool:
        i = 0
        j = 0
        while i < len(word) and j < len(abbr):
            if abbr[j].isalpha():
                if word[i] == abbr[j]:
                    i += 1
                    j += 1
                else:
                    return False
            elif abbr[j].isdigit():
                if abbr[j] == '0':
                    return False
                start = j
                while j < len(abbr) and abbr[j].isdigit():
                    j += 1
                digit = abbr[start : j]
                print(digit)
                n = int(digit)
                i += n
                
        if i == len(word) and j == len(abbr):
            return True
        return False

    # LC 50. Pow(x, n)
    # TC: O(logN), SC: O(logN). The max depth of recursive stack is logN.
    def myPow(self, x: float, n: int) -> float:
        def helper(x, n):
            if n == 0:
                return 1
            if n < 0:
                return helper(1/x, -n)
            if n % 2 == 1:
                tmp = helper(x, (n - 1) // 2)
                return x * tmp * tmp
            if n % 2 == 0:
                tmp = helper(x, n // 2)
                return tmp * tmp
        return helper(x, n)

    # LC 56. Merge Intervals
    # TC: O(NlogN), SC: O(logN) or O(N), quick sort
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort()
        res = [intervals[0]]
        for i in range(1, len(intervals)):
            if intervals[i][0] <= res[-1][1]:
                res[-1][1] = max(res[-1][1], intervals[i][1])
            else:
                res.append(intervals[i])
        return res

    # LC 34. Find First and Last Position of Element in Sorted Array
    # TC: O(logN), SC: O(1)
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def find_first(nums, begin, end, target):
            left = begin
            right = end
            while left < right:
                mid = left + (right - left) // 2
                if nums[mid] < target:
                    left = mid + 1
                elif nums[mid] == target:
                    right = mid
                else:
                    right = mid - 1
            
            if nums[left] == target:
                return left
            return -1
        
        def find_last(nums, begin, end, target):
            left = begin
            right = end
            while left < right - 1:
                mid = left + (right - left) // 2
                if nums[mid] > target:
                    right = mid - 1
                elif nums[mid] == target:
                    left = mid
                else:
                    left = mid + 1
            if nums[right] == target:
                return right
            elif nums[left] == target:
                return left
            else:
                return -1
        
        if not nums:
            return [-1, -1]
        first = find_first(nums, 0, len(nums) - 1, target)
        last = find_last(nums, 0, len(nums) - 1, target)
        return [first, last]

    # LC 1091. Shortest Path in Binary Matrix 
    # TC: O(N) SC: O(1)
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

    # LC 199. Binary Tree Right Side View
    # TC: O(N), SC: O(N)
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        q = deque()
        res = []
        if root:
            q.append(root)
        node = None
        while q:
            for i in range(len(q)):
                node = q.popleft()
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res.append(node.val)
        return res

    # LC Basic Calculator II
    # TC: O(N), SC: O(N)
    def calculate(self, s: str) -> int:
        stack = deque()
        pre_sign = "+"
        num = 0
        i = 0
        while i < len(s):
            if s[i].isdigit():
                num = num * 10 + int(s[i])
            if i == len(s) - 1 or s[i] in "+-*/":
                if pre_sign == "+":
                    stack.append(num)
                elif pre_sign == "-":
                    stack.append(-num)
                elif pre_sign == "*":
                    stack.append(stack.pop() * num)
                elif pre_sign == "/":
                    stack.append(int(stack.pop() / num))
                pre_sign = s[i]
                num = 0
            i += 1
        return sum(stack)

    # LC 347. Top K Frequent Elements
    # TC: O(Nlogk), SC: O(k)
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        counts = Counter(nums)
        pq = []
        for val, cnt in counts.items():
            heapq.heappush(pq, (cnt, val))
            if len(pq) > k:
                heapq.heappop(pq)
        return [x[1] for x in pq]


    # LC 339. Nested List Weight Sum
    # TC: O(N), SC: O(D). N is the total number of nested elements in the input list. D is the max depth of the input list
    def depthSum(self, nestedList: List[NestedInteger]) -> int:
        def helper(nested_list, depth):
            total = 0
            for nested in nested_list:
                if nested.isInteger():
                    total += nested.getInteger() * depth
                else:
                    total += helper(nested.getList(), depth + 1)
            return total
        
        return helper(nestedList, 1)

    # LC 560. Subarray Sum Equals K
    # TC: O(N), SC: O(N)
    # prefix sum + hashmap
    def subarraySum(self, nums: List[int], k: int) -> int:
        sum_dict = defaultdict(int)
        count = 0
        sums = [0] * (len(nums))
        pre_sum = 0
        sum_dict[pre_sum] += 1
        for i in range(len(nums)):
            sums[i] = pre_sum + nums[i]
            pre_sum = sums[i]
            if sums[i] - k in sum_dict:
                count += sum_dict[sums[i] - k]
            sum_dict[pre_sum] += 1
        return count

    # LC 20. Valid Parentheses
    # TC: O(N), SC: O(N)
    def isValid(self, s: str) -> bool:
        stack = deque()
        for c in s:
            if c in "([{":
                stack.append(c)
            elif c == ')':
                if stack and stack.pop() == '(':
                    continue
                else:
                    return False
            elif c == "]":
                if stack and stack.pop() == "[":
                    continue
                else:
                    return False
            elif c == "}":
                if stack and stack.pop() == "{":
                    continue
                else:
                    return False
        return True if len(stack) == 0 else False

    # LC 150. Evaluate Reverse Polish Notation
    # TC: O(N), SC: O(N),
    def evalRPN(self, tokens: List[str]) -> int:
        stack = deque()
        for cur in tokens:
            if cur in "+-*/":
                x2 = stack.pop()
                x1 = stack.pop()
                if cur == "+":
                    res = x1 + x2
                elif cur == "-":
                    res = x1 - x2
                elif cur == "*":
                    res = x1 * x2
                elif cur == "/":
                    res = int(x1 / x2)
                stack.append(res)
            else:
                stack.append(int(cur))
        return stack[0]  

    # LC 739. Daily Temperatures
    # TC: O(N), SC: O(N)
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        if not temperatures:
            return []
        n = len(temperatures)
        res = [0] * n
        stack = [] # save the index of temperature that has not found a warmer one.
        for i, temp in enumerate(temperatures):
            while stack and temp > temperatures[stack[-1]]:
                pre_idx = stack.pop()
                res[pre_idx] = i - pre_idx
            stack.append(i)
        return res

    # LC 853. Car Fleet
    # TC: O(NlogN), SC: O(N)
    def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
        
        cars = sorted(zip(position, speed), reverse=True)
        time = [(target - p) / s for p, s in cars]
        stack = []
        for t in time:
            if not stack or t > stack[-1]:
                stack.append(t)
        return len(stack)

    # LC 994. Rotting Oranges
    # TC: O(N), SC: O(1)
    # BFS
    def orangesRotting(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return -1
        que = deque()
        fresh = 0
        m = len(grid)
        n = len(grid[0])
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    que.append((i, j))
                elif grid[i][j] == 1:
                    fresh += 1
        res = 0
        dirs = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        while que and fresh > 0:
            for i in range(len(que)):
                x, y = que.popleft()
                for dx, dy in dirs:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == 1:
                        que.append((nx, ny))
                        grid[nx][ny] = 2
                        fresh -= 1
            if len(que):
                res += 1

        return res if fresh == 0 else -1

    # LC 402. Remove K Digits
    # TC: O(N), SC: O(N)
    # stack
    def removeKdigits(self, num: str, k: int) -> str:
        stack = []
        remain = len(num) - k
        for digit in num:
            while k and stack and stack[-1] > digit:
                stack.pop()
                k -= 1
            stack.append(digit)
        res = ''.join(stack[:remain]).lstrip('0')
        return res if res else '0'

    # LC 316. Remove Duplicate Letters
    # TC: O(N), SC: O(N)
    # stack
    def removeDuplicateLetters(self, s: str) -> str:
        stack = []
        remain = Counter(s)
        in_stack = set()

        for c in s:
            if c not in in_stack:
                while stack and c < stack[-1] and remain[stack[-1]] > 0:
                    in_stack.remove(stack.pop())
                stack.append(c)
                in_stack.add(c)
            remain[c] -= 1
        return ''.join(stack)

    # LC 74. Search a 2D Matrix
    # TC: O(logN), SC: O(N)
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if matrix is None or len(matrix) == 0 or len(matrix[0]) == 0:
            return False
        m = len(matrix)
        n = len(matrix[0])
        left = 0
        right = m * n - 1
        while left <= right:
            mid = left + (right - left) // 2
            row = mid // n
            col = mid % n
            if matrix[row][col] < target:
                left = mid + 1
            elif matrix[row][col] > target:
                right = mid - 1
            else:
                return True
        return False

    # LC 153. Find Minimum in Rotated Sorted Array
    # TC: O(N), SC: O(1)
    # compare mid with the right element, and we can conclude whether the min value is on the left or right of mid
    def findMin(self, nums: List[int]) -> int:
        left = 0
        right = len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] <= nums[right]:
                right = mid
            elif nums[mid] > nums[right]:
                left = mid + 1
        return nums[left]

    # LC 
    # TC: O(log(N+M)), SC: O(logk)
    # recursive
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        def kth_helper(A, a_left, B, b_left, k):
            if a_left >= len(A):
                return B[b_left + k - 1]
            elif b_left >= len(B):
                return A[a_left + k - 1]
            elif k == 1:
                return min(A[a_left], B[b_left])
            
            a_half_idx = a_left + k // 2 - 1
            a_half_val = A[a_half_idx] if a_half_idx < len(A) else float("inf")

            b_half_idx = b_left + k // 2 - 1
            b_half_val = B[b_half_idx] if b_half_idx < len(B) else float("inf")

            if a_half_val < b_half_val:
                return kth_helper(A, a_half_idx + 1, B, b_left, k - k // 2)
            else:
                return kth_helper(A, a_left, B, b_half_idx + 1, k - k // 2)
        
        total = len(nums1) + len(nums2)
        if total % 2:
            return kth_helper(nums1, 0, nums2, 0, total // 2 + 1)
        else:
            return (kth_helper(nums1, 0, nums2, 0, total // 2 ) + kth_helper(nums1, 0, nums2, 0, total // 2 + 1)) / 2
    # iterative SC: O(1)
    def kth(A, B, k):
            a_left, b_left = 0, 0
            while a_left < len(A) and b_left < len(B) and k > 1:
                a_half_idx = a_left + k // 2 - 1
                a_half_val = A[a_half_idx] if a_half_idx < len(A) else float("inf")
                b_half_idx = b_left + k // 2 - 1
                b_half_val = B[b_half_idx] if b_half_idx < len(B) else float("inf")

                if a_half_val < b_half_val:
                    a_left = a_half_idx + 1
                else:
                    b_left = b_half_idx + 1
                k -= k//2
            if a_left >= len(A):
                return B[b_left + k - 1]
            elif b_left >= len(B):
                return A[a_left + k - 1]
            return min(A[a_left], B[b_left])
        
        total = len(nums1) + len(nums2)
        if total % 2:
            return kth(nums1, nums2, total // 2 + 1)
        else:
            first = kth(nums1, nums2, total // 2)
            second = kth(nums1, nums2, total //2 + 1)
            return (first + second) / 2

    # LC 206. Reverse Linked List
    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, val=0, next=None):
    #         self.val = val
    #         self.next = next
    # Iterative TC: O(N), SC: O(1)
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        dummy.next = None
        cur = head
        while cur:
            nxt = cur.next
            cur.next = dummy.next
            dummy.next = cur
            cur = nxt
        return dummy.next
    # Recursive TC: O(N), SC: O(N)
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        def helper(head):
            if not head or not head.next:
                return head
            new_head = helper(head.next)
            head.next.next = head
            head.next = None
            return new_head
        return helper(head)

    # LC 21. Merge Two Sorted Lists
    # TC: O(N+M), SC: O(1)
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        cur = dummy
        while list1 and list2:
            if list1.val <= list2.val:
                cur.next = list1
                list1 = list1.next
            else:
                cur.next = list2
                list2 = list2.next
            cur = cur.next
        if list1:
            cur.next = list1
        elif list2:
            cur.next = list2
        return dummy.next

    # LC 141. Linked List Cycle
    # TC: O(N), SC: O(1)
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if not head or not head.next:
            return False
        fast, slow = head, head
        while fast.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                return True
        return False

    # LC 19. Remove Nth Node From End of List
    # TC: O(N), SC: O(1)
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode()
        dummy.next = head
        cur = head
        total = 0
        while cur:
            total += 1
            cur = cur.next
        cur = dummy
        for i in range(total - n):
            cur = cur.next
        cur.next = cur.next.next
        return dummy.next

    # LC 138. Copy List with Random Pointer
    # TC: O(N), SC: O(N)
    # Iterative
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head:
            return None
        visited = {}
        cur = head
        while cur:
            if cur not in visited:
                cur_copy = Node(cur.val)
                visited[cur] = cur_copy
            cur_copy = visited[cur]
            if cur.next:
                if cur.next not in visited:
                    next_copy = Node(cur.next.val)
                    visited[cur.next] = next_copy
                cur_copy.next = visited[cur.next]
            if cur.random:
                if cur.random not in visited:
                    rand_copy = Node(cur.random.val)
                    visited[cur.random] = rand_copy
                cur_copy.random = visited[cur.random]

            cur = cur.next
        return visited[head]
    # Recursive
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        def dfs(head, visited):
            if not head:
                return head
            if head in visited:
                return visited[head]
            cur_copy = Node(head.val)
            visited[head] = cur_copy
            next_copy = dfs(head.next, visited)
            cur_copy.next = next_copy
            rand_copy = dfs(head.random, visited)
            cur_copy.random = rand_copy

            return cur_copy

        visited = {}
        return dfs(head, visited)

    # LC 2. Add Two Numbers
    # TC: O(max(M, N)), SC:O(Max(M, N))
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        cur = dummy
        carry = 0
        while l1 and l2:
            cur_sum = l1.val + l2.val + carry
            node = ListNode(cur_sum % 10)
            carry = cur_sum // 10
            cur.next = node
            cur = cur.next
            l1 = l1.next
            l2 = l2.next
        while l1:
            cur_sum = l1.val + carry
            node = ListNode(cur_sum % 10)
            carry = cur_sum // 10
            cur.next = node
            cur = cur.next
            l1 = l1.next
        while l2:
            cur_sum = l2.val + carry
            node = ListNode(cur_sum % 10)
            carry = cur_sum // 10
            cur.next = node
            cur = cur.next
            l2 = l2.next
        if carry:
            node = ListNode(carry)
            cur.next = node
        return dummy.next

    # LC 23. Merge k Sorted Lists
    # TC: O(N), SC: O(k)
    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, val=0, next=None):
    #         self.val = val
    #         self.next = next
    class NodeCmp:
        def __init__(self, node):
            self.node = node
        
        def __lt__(self, other):
            return self.node.val < other.node.val

    class Solution:    
        def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
            dummy = ListNode()
            pq = []
            for i in range(len(lists)):
                if lists[i]:
                    heapq.heappush(pq, NodeCmp(lists[i]))
            cur = dummy
            while pq:
                node_cmp = heapq.heappop(pq)
                cur.next = node_cmp.node
                cur = cur.next
                if cur.next:
                    heapq.heappush(pq, NodeCmp(cur.next))
            return dummy.next
        
    # LC 25. Reverse Nodes in k-Group
    # TC: O(N), SC:O(n/k)
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        def dfs(head, k):
            cur = head
            for i in range(k):
                if not cur:
                    return head
                cur = cur.next
            dummy = ListNode(0)
            dummy.next = dfs(cur, k)
            for i in range(k):
                nxt = head.next
                head.next = dummy.next
                dummy.next = head
                head = nxt
            return dummy.next
        return dfs(head, k)
        
