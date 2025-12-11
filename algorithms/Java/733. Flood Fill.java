class Solution {
    public int[][] floodFill(int[][] image, int sr, int sc, int color) {
        if (image == null || image.length == 0 || image[0].length == 0) {
            return image;
        }
        int m = image.length;
        int n = image[0].length;
        if (image[sr][sc] == color) {
            return image;
        }
        int[][] dirs = {{-1, 0}, {1, 0}, {0, 1}, {0, -1}};
        Queue<Pair> queue = new LinkedList<>();
        int preColor = image[sr][sc];
        queue.offer(new Pair(sr, sc));
        while (!queue.isEmpty()) {
            Pair top = queue.poll();
            image[top.x][top.y] = color;
            for (int[] dir : dirs) {
                int newX = top.x + dir[0];
                int newY = top.y + dir[1];
                if (newX >= 0 && newX < m && newY >=0 && newY < n && image[newX][newY] == preColor) {
                    queue.offer(new Pair(newX, newY));
                }
            }
        }
        return image;
    }
}
class Pair {
    int x;
    int y;
    public Pair(int x, int y) {
        this.x = x;
        this.y = y;
    }
}