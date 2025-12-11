import java.util.HashMap;
import java.util.Map;

class Solution {
    public boolean isAnagram(String s, String t) {
        if (s == null && t == null){
            return true;
        }else if (s == null || t == null){
            return false;
        }else if (s.length() != t.length()){
            return false;
        }
        Map<Character, Integer> maps = new HashMap<>();
        Map<Character, Integer> mapt = new HashMap<>();
        for (int i = 0; i < s.length(); i++){
            maps.put(s.charAt(i), maps.getOrDefault(s.charAt(i), 0) + 1);
        }
        for (int j = 0; j < t.length(); j++){
            mapt.put(t.charAt(j), mapt.getOrDefault(t.charAt(j), 0) + 1);
        }
        return maps.equals(mapt);
    }
}