class Solution {
    public int[] findBuildings(int[] heights) {
        List<Integer> list = new ArrayList<>();
        int max_height = -1;
        for (int i = heights.length - 1; i >= 0; i--){
            if (max_height < heights[i]){
                list.add(i);
                max_height = heights[i];
            }
        }
        int[] res = new int[list.size()];
        for(int i = 0; i < list.size(); i++){
            res[i] = list.get(list.size() - i - 1);
        }
        return res;
    }
}