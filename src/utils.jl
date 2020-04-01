function partition(x::Array, chunks::Array{Int})
    _iend = cumsum(chunks)
    IDX = map((_start, _end) -> _start:_end, [1; _iend[1:end-1].+1], _iend) # Get the index
end
