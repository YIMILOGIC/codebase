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
            
