class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int currSum = nums[0];
        int maxSum = nums[0];

        for (int i = 1; i < nums.size(); i++) {

            if (currSum < 0) currSum = nums[i];
            else currSum += nums[i];
            
            maxSum = std::max(maxSum, currSum);
        }

        return maxSum;
    }
};