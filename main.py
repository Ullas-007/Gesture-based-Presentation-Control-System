class Solution():
    def twoSum(self, nums, target):
        for i in nums:
            for j in nums:
                if i+j == target and i != j:
                    return [nums.index(i), nums.index(j)]
obj = Solution()
n = eval(input(""))
t = int(input(""))
a=obj.twoSum(n, t)
print(a)
