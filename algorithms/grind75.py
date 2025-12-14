class Solution:
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

    # LC 20. Valid Parentheses
    # TC: O(N), SC: O(N)
    def isValid(self, s: str) -> bool:
        stack = deque()
        for c in s:
            if c in "{([":
                stack.append(c)
                continue
            elif c == ")" and stack and stack.pop() == "(":
                continue
            elif c == "}" and stack and stack.pop() == "{":
                continue
            elif c == "]" and stack and stack.pop() == "[":
                continue
            else:
                return False
        return True if len(stack) == 0 else False

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

    # LC 121. Best Time to Buy and Sell Stock
    # TC: O(N), SC: O(1)
    def maxProfit(self, prices: List[int]) -> int:
        min_price = prices[0]
        res = 0
        for price in prices:
            res = max(price - min_price, res)
            min_price = min(min_price, price)
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
    # LC 226. Invert Binary Tree
    # TC: O(N), SC: O(Height)
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def dfs(root):
            if root is None:
                return
            root.left, root.right = root.right, root.left
            dfs(root.left)
            dfs(root.right)
        dfs(root)
        return root
    # LC 242. Valid Anagram
    # TC: O(M+N), SC: O(1) The number of characters is limited (26).
    def isAnagram(self, s: str, t: str) -> bool:
        src_dict = defaultdict(int)
        tgt_dict = defaultdict(int)
        for c in s:
            src_dict[c] += 1
        for c in t:
            tgt_dict[c] += 1
        return src_dict == tgt_dict

    def isAnagram2(self, s: str, t: str) -> bool:
        if s is None and t is None:
            return True
        elif s is None or t is None:
            return False
        elif len(s) != len(t):
            return False
        
        counter = DefaultDict(int)
        for c in s:
            counter[c] += 1
        for c in t:
            counter[c] -= 1
        for c in counter:
            if counter[c] != 0:
                return False
        return True
    # 704. Binary Search
    # TC: O(logN), SC: O(1)
    def search(self, nums: List[int], target: int) -> int:
        if not nums:
            return -1
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1

    # 733. Flood Fill
    # TC: O(N), SC: O(N)
    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        if not image or not image[0]:
            return image
        m = len(image)
        n = len(image[0])
        if image[sr][sc] == color:
            return image
        queue = deque()
        queue.append((sr, sc))
        pre_color = image[sr][sc]
        DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        while queue:
            cur_r, cur_c = queue.popleft()
            image[cur_r][cur_c] = color
            for dr, dc in DIRECTIONS:
                new_r, new_c = cur_r + dr, cur_c + dc
                if 0 <= new_r < m and 0 <= new_c < n and image[new_r][new_c] == pre_color:
                    queue.append((new_r, new_c))
        return image

    # 235. Lowest Common Ancestor of a Binary Search Tree
    # TC: O(N), SC: O(H)
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        def dfs(root, p, q):
            if not root:
                return None
            if p.val < root.val and q.val < root.val:
                return dfs(root.left, p, q)
            elif p.val > root.val and q.val > root.val:
                return dfs(root.right, p, q)
            else:
                return root
        return dfs(root, p, q)

    # 110. Balanced Binary Tree
    # TC: O(N), SC: O(H)
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def dfs(root):
            if not root:
                return True, 0
            left_balanced, left_height = dfs(root.left)
            right_balanced, right_height = dfs(root.right)
            cur_height = max(left_height, right_height) + 1
            cur_balanced = left_balanced and right_balanced and abs(left_height - right_height) < 2
            return cur_balanced, cur_height
        return dfs(root)[0]

