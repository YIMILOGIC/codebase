class Solution {
    public int[] sortArray(int[] nums) {
        int[] temp = new int[nums.length];
        mergeSort(nums, 0, nums.length - 1, temp);
        return nums;
    }

    private void mergeSort(int[] nums, int left, int right, int[] temp){
        if (left >= right){
            return;
        }
        int mid = left + (right - left) / 2;
        mergeSort(nums, left, mid, temp);
        mergeSort(nums, mid + 1, right, temp);
        int i = left, j = mid + 1, k = left;
        while (i <= mid && j <= right){
            if (nums[i] <= nums[j]){
                temp[k] = nums[i];
                i++;
            }else {
                temp[k] = nums[j];
                j++;
            }
            k++;
        }
        while (i <= mid){
            temp[k] = nums[i];
            i++;
            k++;
        }
        while (j <= right){
            temp[k] = nums[j];
            j++;
            k++;
        }
        for (i = left; i <= right; i++){
            nums[i] = temp[i];
        }
            
    }
}