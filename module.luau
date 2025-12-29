--[=[
	-- Weighted Random --
	
	A library that provides functions and methods to sample from a discrete probability distribution, with
	various algorithms suited for specific scenarios.
	
	Copyright (c) 2025 Nowoshire
	
	This software is licensed under the MIT License.
		https://github.com/Nowoshire/Luau-Weighted-Random/blob/main/LICENSE
		
	Available Algorithms:
		- Linear Scan
			- Very simple algorithm with minimal or no precomputation, at the cost of O(n) sampling.
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
		> Linear Scan <
		---------------
		--[[
			Creates a new LinearTable from the given weights for the Linear Scan method.
			This is an optional precomputation step that can be used to avoid recalculating the total weight if the distribution has not changed.
			
			The complexity of this function is O(n).
		]]
		function buildLinear <T>(weights: T): (LinearTable<T>)
	
		--[[
			Picks a random value from the given LinearTable using a linear scan.
			The complexity of this function is O(n).
		]]
		function sampleLinear <T>(linearTable: LinearTable<T>): (keyof<T>)
		
		--[[
			Picks a random value from the given weights using a linear scan.
			The complexity of this function is O(n).
		]]
		function WeightedRandom.sampleLinearNoPrecomp <T>(weights: T): (keyof<T>)
		
		[ LinearTable ]:
			--[[
				Picks a random value from the given LinearTable using a linear scan.
				The complexity of this function is O(n).
			]]
			function Sample <T>(self: LinearTable<T>): (keyof<T>) 
		
		> Prefix Sum + Binary Search <
		------------------------------
		--[[
			Creates a new PreSumTable from the given weights for the Prefix Sum + Binary Search method.
			The complexity of this function is O(n).
		]]
		function buildPreSum <T>(weights: T): (PreSumTable<T>)
		
		--[[
			Picks a random value from the given PreSumTable using a binary search.
			The complexity of this function is O(log n).
		]]
		function samplePreSum <T>(presumTable: PreSumTable<T>): (keyof<T>)
		
		[ PreSumTable ]:
			--[[
				Picks a random value from the given PreSumTable using a binary search.
				The complexity of this function is O(log n).
			]]
			function Sample <T>(self: PreSumTable<T>): (keyof<T>)
		
		> Alias Method (A.J. Walker) <
		------------------------------
		--[[
			Creates a new AliasTable from the given weights for the Alias Method by A.J. Walker.
			The complexity of this function is O(n).
		]]
		function buildAlias <T>(weights: T): (AliasTable<T>)
	
		--[[
			Picks a random value from the given AliasTable using a constant-time search.
			The complexity of this function is O(1).
		]]
		function sampleAlias <T>(aliasTable: AliasTable<T>): (keyof<T>)
		
		[ AliasTable ]:
			--[[
				Picks a random value from the given AliasTable using a constant-time search.
				The complexity of this function is O(1).
			]]
			function Sample <T>(self: AliasTable<T>): (keyof<T>)
]=]

--!strict
--!optimize 2
--!native

local WeightedRandom = {}

-- Misc --
local rng = Random.new()

--[[
	Returns the sum of all the weights in the given table.
]]
local function sumWeight(weights: {number}): (number)
	local sum = 0
	for _, weight in weights do
		sum += weight
	end

	return sum
end

--[[
	Builds a weight array and a lookup table from the given <code>weight</code> table if it contains a hash portion, otherwise returns the original <code>weight</code> table and nil.
]]
local function buildWeightArrayAndLookup <K>(weights: WeightTable<K>): ({number}, {K}?)
	local numWeights = #weights
	local lastKey, lastWeight = next(weights, if numWeights ~= 0 then numWeights else nil)

	-- Check if the weight table has a hash portion and skip conversion if it doesn't.
	-- Luau guarantees that the array portion of a table is visited before the hash portion, and the element at
	-- index #t is stored in the array portion, meaning we can do a quick amortized O(1) check with `next(t, #t)`.
	-- See https://rfcs.luau.org/generalized-iteration.html#default-table-iteration for more information.
	if lastKey == nil then
		return weights, nil
	end

	numWeights += 1

	-- weights has a hash portion, build weight array and lookup table.
	local weightArray = table.move(weights, 1, numWeights, 1, table.create(numWeights) :: {number}) -- prealloc what we can
	local lookup = {
		[numWeights] = lastKey	
	} -- only provides lookup for entries in the hash portion
	weightArray[numWeights] = lastWeight

	for key, weight in next, weights, lastKey do
		numWeights += 1
		weightArray[numWeights] = weight
		lookup[numWeights] = key
	end

	return weightArray, lookup
end

-- Library Functions --

local linearMetatable = {
	__index = {
		Sample = WeightedRandom.sampleLinear
	}
}

--[[
	Creates a new <code>LinearTable</code> from the given <code>weights</code> for the <strong>Linear Scan</strong> method.
	This is an optional precomputation step that can be used to avoid recalculating the total weight if the distribution has not changed.
	
	The complexity of this function is <strong><em>O(n)</em></strong>.
]]
function WeightedRandom.buildLinear <T>(weights: T): (LinearTable<T>)
	return setmetatable({
		weights = weights,
		sum = sumWeight(weights),
	}, linearMetatable)
end

--[[
	Picks a random value from the given <code>LinearTable</code> using a linear scan.
	The complexity of this function is <strong><em>O(n)</em></strong>.
]]
function WeightedRandom.sampleLinear <T>(linearTable: LinearTable<T>): (keyof<T>)
	local total = linearTable.sum
	local weights = linearTable.weights
	local rand = rng:NextNumber(0, total)

	for k, weight in weights do
		rand -= weight

		if rand <= 0 then
			return k
		end
	end

	-- fallback
	return (next(weights))
end

--[[
	Picks a random value from the given <code>weights</code> using a linear scan.
	The complexity of this function is <strong><em>O(n)</em></strong>.
]]
function WeightedRandom.sampleLinearNoPrecomp <T>(weights: T): (keyof<T>)
	weights = weights :: WeightTable -- cant annotate the types of generics, so we'll have to do this
	
	local total = sumWeight(weights)
	local rand = rng:NextNumber(0, total)

	for k, weight in weights do
		rand -= weight

		if rand <= 0 then
			return k
		end
	end

	-- fallback
	return (next(weights))
end

local presumMetatable = {
	__index = {
		Sample = WeightedRandom.samplePreSum
	}
}

--[[
	Creates a new <code>PreSumTable</code> from the given <code>weights</code> for the <strong>Prefix Sum + Binary Search</strong> method.
	The complexity of this function is <strong><em>O(n)</em></strong>.
]]
function WeightedRandom.buildPreSum <T>(weights: T): (PreSumTable<T>)
	local weightArray, lookup = buildWeightArrayAndLookup(weights)
	local numWeights = #weightArray
	local prefixSums = table.create(numWeights) :: {number} -- pre-alloc
	local weightSum = 0

	for i, weight in weightArray do
		weightSum += weight
		prefixSums[i] = weightSum
	end

	return setmetatable({
		preSums = prefixSums,
		sum = weightSum,
		n = numWeights,
		lookup = lookup
	}, presumMetatable)
end

--[[
	Picks a random value from the given <code>PreSumTable</code> using a binary search.
	The complexity of this function is <strong><em>O(log n)</em></strong>.
]]
function WeightedRandom.samplePreSum <T>(presumTable: PreSumTable<T>): (keyof<T>)
	local prefixSums = presumTable.preSums
	local low, high = 1, presumTable.n
	local rand = rng:NextNumber(0, presumTable.sum)
	local lookup = presumTable.lookup

	-- Binary search
	while low < high do
		local mid = (low + high) // 2
		if rand > prefixSums[mid] then
			low = mid + 1
		else
			high = mid
		end
	end

	return lookup and lookup[low] or low :: any
end

local aliasMetatable = {
	__index = {
		Sample = WeightedRandom.sampleAlias
	}
}

--[[
	Creates a new <code>AliasTable</code> from the given <code>weights</code> for the <strong>Alias Method</strong> by A.J. Walker.
	The complexity of this function is <strong><em>O(n)</em></strong>.
]]
function WeightedRandom.buildAlias <T>(weights: T): (AliasTable<T>)
	local weightSum = sumWeight(weights)
	local weightArray, lookup = buildWeightArrayAndLookup(weights)
	local numWeights = #weightArray
	local avgNormWeight = numWeights / weightSum
	local normWeights = table.create(numWeights) :: {number}

	-- pre-alloc, optimal heuristic for all scenarios, avoids expensive rehashing down the line.
	local smallStack, largeStack = table.create(numWeights) :: {number}, table.create(numWeights) :: {number}

	for i, weight in weightArray do
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

	return setmetatable({
		alias = alias,
		prob = prob,
		n = numWeights,
		lookup = lookup
	}, aliasMetatable)
end

--[[
	Picks a random value from the given <code>AliasTable</code> using a constant-time search.
	The complexity of this function is <strong><em>O(1)</em></strong>.
]]
function WeightedRandom.sampleAlias <T>(aliasTable: AliasTable<T>): (keyof<T>)
	local rand = rng:NextInteger(1, aliasTable.n)
	local result = if rng:NextNumber() < aliasTable.prob[rand]
		then rand
		else aliasTable.alias[rand]
	local lookup = aliasTable.lookup

	return lookup and lookup[result] or result :: any
end

-- Types --
type WeightTable<K = any> = {
	[K]: number
}

export type LinearTable<weights = WeightTable> = typeof(setmetatable({} :: {
	weights: WeightTable,
	sum: number
}, linearMetatable))

export type PreSumTable<weights = WeightTable> = typeof(setmetatable({} :: {
	preSums: {number},
	sum: number,
	n: number,
	lookup: {any}?
}, presumMetatable))

export type AliasTable<weights = WeightTable> = typeof(setmetatable({} :: {
	alias: {number},
	prob: {number},
	n: number,
	lookup: {any}?
}, aliasMetatable))

return WeightedRandom
