--[[
	-- Weighted Random --
	
	A library that provides functions to sample from a discrete probability distribution, with
	various algorithms suited for specific scenarios.
	
	Copyright (c) 2025 Nowoshire
	
	This software is licensed under the MIT License.
		https://github.com/Nowoshire/Luau-Weighted-Random/blob/main/LICENSE
		
	Available Algorithms:
		- Linear Scan
			- Very simple algorithm with optional or no precomputation, at the cost of O(n) sampling.
			- Trivial dynamic weights.
			- Suited for small or ephemeral distributions for a handful of samples.
			
		- Prefix Sum + Binary Search
			- Simpler precomputation than Alias Method at the cost of O(log n) sampling.
			- Currently does not support dynamic updates but there are plans to support it in the future
			  with Fenwick/Binary Indexed Trees.
			- Suited for a moderate number of samples or changing distributions.
	
		- Alias Method (A.J. Walker)
			- Provides constant-time O(1) sampling at the cost of complex O(n) precomputation and setup.
			- Does not support dynamic updates (requires rebuilding).
			- Suited for heavy sampling with static weights.
	
	Library:
		[ Linear Scan ]:
			type LinearTable<K = any> = {
				weights: {[K]: number},
				sum: number,
				n: number
			}
		
			-- Creates a new LinearTable from the given weights for the Linear Scan method.
			-- This is an optional precomputation step that can be used to avoid recalculating the total weight if the distribution has not changed.
			--
			-- The complexity of this function is O(n).
			function newLinear <K>(weights: {[K]: number}): (LinearTable<K>)
		
			-- Picks a random value from the given LinearTable or table of weights using a linear scan.
			-- The complexity of this function is O(n).
			function pickLinear <K>(linearTable: LinearTable<K> | {[K]: number}): (K | number)
			
		[ Prefix Sum + Binary Search ]:
			type PreSumTable = {
				preSums: {number},
				sum: number,
				n: number
			}
			
			-- Creates a new PreSumTable from the given weights for the Prefix Sum + Binary Search method.
			-- The complexity of this function is O(n).
			function newPreSum(weights: {number}): (PreSumTable)
			
			-- Picks a random index from the given PreSumTable using a binary search.
			-- The complexity of this function is O(log n).
			function pickPreSum(presumTable: PreSumTable): (number)
		
		[ Alias Method (A.J. Walker) ]:
			type AliasTable = {
				alias: {number},
				prob: {number},
				n: number
			}
		
			-- Creates a new AliasTable from the given weights for the Alias Method by A.J. Walker.
			-- The complexity of this function is O(n).
			function newAlias(weights: {number}): (AliasTable)
		
			-- Picks a random index from the given AliasTable using a constant-time search.
			-- The complexity of this function is O(1).
			function pickAlias(aliasTable: AliasTable): (number)
]]

--!strict
--!optimize 2

local WeightedRandom = {}

-- Types --
export type AliasTable = {
	alias: {number},
	prob: {number},
	n: number
}

export type PreSumTable = {
	preSums: {number},
	sum: number,
	n: number
}

export type LinearTable<K = any> = {
	weights: {[K]: number},
	sum: number,
	n: number
}

-- Misc --
local rng = Random.new()

local function sumWeight(weights: {number}): (number)
	local sum = 0
	for _, weight in weights do
		sum += weight
	end
	
	return sum
end

-- Library Functions --

--[[
	Creates a new <code>LinearTable</code> from the given <code>weights</code> for the <strong>Linear Scan</strong> method.
	This is an optional precomputation step that can be used to avoid recalculating the total weight if the distribution has not changed.
	
	The complexity of this function is <strong><em>O(n)</em></strong>.
]]
function WeightedRandom.newLinear <K>(weights: {[K]: number}): (LinearTable<K>)
	return {
		weights = weights,
		sum = sumWeight(weights),
		n = #weights
	}
end

--[[
	Picks a random value from the given <code>LinearTable</code> or table of weights using a linear scan.
	The complexity of this function is <strong><em>O(n)</em></strong>.
]]
function WeightedRandom.pickLinear <K>(linearTable: LinearTable<K> | {[K]: number}): (K)
	local total = (linearTable :: LinearTable).sum or sumWeight(linearTable)
	local weights = (linearTable :: LinearTable).weights or linearTable
	local rand = rng:NextNumber() * total
	
	for i, weight in weights do
		rand -= weight
		
		if rand <= 0 then
			return i
		end
	end
	
	return (linearTable :: LinearTable).n or #weights
end

--[[
	Creates a new <code>PreSumTable</code> from the given <code>weights</code> for the <strong>Prefix Sum + Binary Search</strong> method.
	The complexity of this function is <strong><em>O(n)</em></strong>.
]]
function WeightedRandom.newPreSum(weights: {number}): (PreSumTable)
	local weightSum = 0
	local numWeights = #weights
	local prefixSums = table.create(numWeights) :: {number} -- pre-alloc
	
	for i, weight in weights do
		weightSum += weight
		prefixSums[i] = weight
	end
	
	return {
		preSums = prefixSums,
		sum = weightSum,
		n = numWeights,
	}
end

--[[
	Picks a random value from the given <code>PreSumTable</code> using a binary search.
	The complexity of this function is <strong><em>O(log n)</em></strong>.
]]
function WeightedRandom.pickPreSum(presumTable: PreSumTable): (number)
	local prefixSums = presumTable.preSums
	local low, high = 1, presumTable.n
	local rand = rng:NextNumber() * presumTable.sum
	
	-- Binary search
	while low < high do
		local mid = (low + high) // 2
		if rand > prefixSums[mid] then
			low = mid + 1
		else
			high = mid
		end
	end
	
	return low
end

--[[
	Creates a new <code>AliasTable</code> from the given <code>weights</code> for the <strong>Alias Method</strong> by A.J. Walker.
	The complexity of this function is <strong><em>O(n)</em></strong>.
]]
function WeightedRandom.newAlias(weights: {number}): (AliasTable)
	local weightSum = sumWeight(weights)

	local numWeights = #weights
	local avgNormWeight = numWeights / weightSum
	local normWeights = table.create(numWeights) :: {number}

	-- pre-alloc, optimal heuristic for all scenarios, avoids expensive reallocations down the line.
	local smallStack, largeStack = table.create(numWeights) :: {number}, table.create(numWeights) :: {number}

	for i, weight in weights do
		local norm = weight * avgNormWeight
		normWeights[i] = norm

		if norm < 1 then
			table.insert(smallStack, i)
		else
			table.insert(largeStack, i)
		end
	end

	-- pre-alloc
	local prob, alias = table.create(numWeights) :: {number}, table.create(numWeights) :: {number}
	while smallStack[1] and largeStack[1] do -- both non-empty
		local small, large = table.remove(smallStack) :: number, table.remove(largeStack) :: number
		prob[small] = normWeights[small]
		alias[small] = large

		local normLarge = normWeights[large] + normWeights[small] - 1
		normWeights[large] = normLarge

		if normLarge < 1 then
			table.insert(smallStack, large)
		else
			table.insert(largeStack, large)
		end
	end

	-- fill any missed values with 1
	for _, v in largeStack do
		prob[v] = 1
	end

	for _, v in smallStack do
		prob[v] = 1
	end

	return {
		alias = alias,
		prob = prob,
		n = numWeights,
	}
end

--[[
	Picks a random value from the given <code>AliasTable</code> using a constant-time search.
	The complexity of this function is <strong><em>O(1)</em></strong>.
]]
function WeightedRandom.pickAlias(aliasTable: AliasTable): (number)
	local indexInt = rng:NextInteger(1, aliasTable.n)

	return if rng:NextNumber() < aliasTable.prob[indexInt]
		then indexInt
		else aliasTable.alias[indexInt]
end

return WeightedRandom
