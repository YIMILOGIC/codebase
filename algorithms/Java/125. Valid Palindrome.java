class Solution {
    public boolean isPalindrome(String s) {
        if (s.isEmpty()){
            return true;
        }
        int left = 0;
        int right = s.length() - 1;
        while (left < right){
            char chLeft = s.charAt(left);
            if (!Character.isLetterOrDigit(chLeft)){
                left++;
                continue;
            }
            char chRight = s.charAt(right);
            if (!Character.isLetterOrDigit(chRight)){
                right--;
                continue;
            }
            if (Character.toLowerCase(chLeft) != Character.toLowerCase(chRight)){
                return false;
            } else{
                left ++;
                right --;
            }
        }
        return true;
    }
}