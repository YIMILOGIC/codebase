/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class NodeInfo {
    boolean isBalanced;
    int height;
    public NodeInfo(boolean isBalanced, int height){
        this.isBalanced = isBalanced;
        this.height = height;
    }
}

class Solution {
    public boolean isBalanced(TreeNode root) {
        return helper(root).isBalanced;
    }
    
    private NodeInfo helper(TreeNode root) {
        if (root == null) {
            return new NodeInfo(true, 0);
        }
        NodeInfo leftInfo = helper(root.left);
        NodeInfo rightInfo = helper(root.right);
        int height = Math.max(leftInfo.height, rightInfo.height) + 1;
        boolean isBalanced = leftInfo.isBalanced && rightInfo.isBalanced && Math.abs(leftInfo.height - rightInfo.height) < 2;
        return new NodeInfo(isBalanced, height);
    }
}
