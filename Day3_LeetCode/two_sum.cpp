#include <iostream>
#include <vector>


class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        
        int temp = target;

        std::unordered_map<int, int> ind;
        std::vector<int> ans;

        for (int i = 0; i < nums.size(); i++) {
            int toFind = target - nums[i];
            if (ind.contains(toFind)) {
                return {i, ind[toFind]};
            }
            ind[nums[i]] = i;
        }        

        return {};
    }
};