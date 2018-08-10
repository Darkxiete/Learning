---
whL	 typora-copy-images-to: pic
---

#1. 排序算法

主要几个问题：

1. 算法的思想、实现
2. 算法的复杂度分析，包括时间复杂度、空间复杂度
3. 算法的应用场景、优缺点



## 1. 排序算法的分类与定义

排序分为内排序和外排序。

内排序的定义如下：

> **内部排序：**指的是待排序记录存放在计算机随机存储器（内存）中进行的排序过程；我们熟悉常用的冒泡排序，选择排序，插入排序，快速排序，堆排序，归并排序，希尔排序……等都属于内部排序方法；

简单理解就是在内存里排序。

外排序的的定义为：

> **外部排序：**指的是待排序记录的数量很大，以致内存一次不能容纳全部记录，在排序过程中需要对外存进行访问的排序过程；

简单来说就是，待排序的数据不能一次性读入内存中。

## 2. 相关介绍

### 1. 主定理

![主定理](pic\主定理.png)

对于归并排序和快速排序(最好情况)：
$$
T(n)=2T(n/2)+O(n) \\
a=2,b=2,f(n)=O(n) \\
\therefore n^{\log _a ^b}=n^{\log_2^2}=n \\
\therefore T(n)=O(n*\log n)=O(n \log n)
$$

### 2. 算法稳定性

概念：通俗地讲就是能保证排序前两个**相等**的数据其在序列中的先后位置顺序与排序后它们两个先后位置顺序相同。即：如，如果$A_i == A _j$，$Ai$ 原来在 $A_j$ 位置前，排序后 $Ai$  仍然是在 $Aj$ 位置前。 

意义：

### 3. 内排序(比较排序)

![排序算法复杂度小结](pic\排序算法复杂度小结.png)



#### 冒泡排序

相邻元素逐个比较，找出最大\小

方案1

此方案中，不管数组是否已经排序，其复杂度都为$\Theta(n^2)$。

```python
test_list = [2, 4, 1, 3, 10, 9, 5]
def bubble_sort(l):
    length = len(l)
    max_idx = length -1 
    for i in range(length):
        for j in range(length-i):
            print("i={},j={}".format(i,j),l)
            if j == max_idx:
                break
            if l[j]>l[j+1]:
                l[j], l[j+1] = l[j+1], l[j]
    return l
bubble_sort(test_list)
```



方案2

如果一次搜索中没有发生交换，则为已排序数组，无需再排序。

```python
test_list = [2, 4, 1, 3, 10, 9, 5]
def bubble_sort(l):
    length = len(l)
    swapped = False
    for i in range(length):
        for j in range(length-i-1):
            print("i={},j={}".format(i,j),l)
            if l[j]>l[j+1]:
                l[j], l[j+1] = l[j+1], l[j]
                swapped = True
    	if not swapped:
        	return l
    return l
bubble_sort(test_list)
```



稳定性：

稳定。相同的两个元素只要不互换，则排序后其顺序保持不变。



#### 插入排序

将未排序的数字与已左边已排序数组中的每个数进行比较，插入

~~方案1~~

```python
test_list = [2, 4, 1, 3, 10, 9, 5]
def insert_sort(l):
    length = len(l)
    for i in range(1,length):
        # 待排序索引
        for j in range(1,i+1)[::-1]:
            if l[j]<l[j-1]:
                l[j],l[j-1]=l[j-1],l[j]
        	print("i: {}, j: {}".format(i,j),l)
    return l
insert_sort(test_list)
```

方案2 increase

重点是在插入操作，先保存要插入的值，然后在待插入数组中逐个进行比较，较大值往前挪一个位置，直到找不到大值(i)，插入此位置(i+1)。

```python
test_list = [2, 4, 1, 3, 10, 9, 5]
def insert_sort(arr):
    length = len(arr)
    for i in range(1,length):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key
    return arr
insert_sort(test_list)
```

方案3 decrease

```python
test_list = [2, 4, 1, 3, 10, 9, 5]
def insert_sort(arr):
    length = len(arr)
    for i in range(1,length):
        key = arr[i]
        j = i - 1
        while j >= 0 and key > arr[j]:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key
    return arr
insert_sort(test_list)
```



优化方法：

插入排序时间复杂度主要体现在查找元素需要插入的正确位置上，在查找位置时使用二分法可以减少每次插入所需比较的次数。

稳定性：
稳定，对于相同的数值，假设A已经在已排序数组中，插入B时，只可能在其右边，保持了顺序。

#### 希尔排序

> 学过InsertSort排序的都应该了解，直接插入排序算法适用于 基本有序和数据量不大的排序序列。基于这两点1959年Donald L. Shell提出了希尔排序，又称“缩小增量排序”。 

希尔排序的思想就是通过逐步建立一个“基本有序”的数组，得到有序数组。

比如：

21,34,36,10,29,40,82,91,51,45,32,56

按照序列[5,3,1]对其排序，首先将其变成一个二维矩阵形式。
$$
\begin{matrix}
21 & 34 & 36 & 10 & 29 \\
40 & 82 & 91 & 51 & 45 \\
32 & 56
  \end{matrix} \tag{1}
$$
然后按列进行排序，得到如下形式
$$
\begin{matrix}
21 & 34 & 36 & 10 & 29 \\
32 & 56 & 91 & 51 & 45 \\
40 & 82
  \end{matrix} \tag{2}
$$
然后将矩阵变成$3*4$形式
$$
\begin{matrix}
21 & 34 & 36 \\
10 & 29 & 40 \\
82 & 91 & 51 \\
45 & 32 & 56
  \end{matrix} \tag{3}
$$
继续按列排序
$$
\begin{matrix}
10 & 29 & 36 \\
21 & 32 & 40 \\
45 & 34 & 51 \\
82 & 91 & 56
  \end{matrix} \tag{4}
$$
最后变成$12*1$的向量，即得到有序数组

```python
def shell_sort(arr):
    n = len(arr)
    gap = n // 2
    while(gap > 0):
        for i in range(gap,n):
            temp = arr[i]
            j = i
            while(j >= gap and arr[j-gap] > temp):
                arr[j] = arr[j-gap]
                j -= gap
            arr[j] = temp	
        gap //= 2
test_list = [2, 4, 1, 3, 10, 9, 5]
shell_sort(test_list)
```



相关参考

> [Shell sort](http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/shell/shellen.htm)



#### 选择排序

每次选出最大\小的

```python
test_list = [2, 4, 1, 3, 10, 9, 5]
# 最大值
def select_sort(l):
    length = len(l)
    for i in range(length):
        max_idx = 0
        max_value = l[max_idx]
        for j in range(length-i):
            print("i={},j={},max_idx={},max_value={}".format(i,j,max_idx,l[max_idx]),l)
            if l[j] > l[max_idx]:
                max_idx = j
        max_value = l[max_idx] 
        # j之后的数 降一个索引值
#        if i ==0:
#        	l[max_idx:-1-i] = l[max_idx+1:]
#        else:
#            l[max_idx:-1-i] = l[max_idx+1:-i]
#        l[length-1-i] = max_value
		l[max_idx], l[length-i-1] = l[length-i-1], l[max_idx]
	return l
select_sort(test_list) 

test_list = [2, 4, 1, 3, 10, 9, 5]
# 最小值
def select_sort(l):
    length = len(l)
    for i in range(length):
        min_idx = i
        min_value = l[i]
        for j in range(i,length):
            if l[j] < l[min_idx]:
                min_idx = j
        min_value = l[min_idx]
        print("i:{}, j:{}, min:{}, arr:{}".format(i,j,min_value,l))
        l[min_idx], l[i] = l[i], l[min_idx]
	return l
select_sort(test_list) 

```





稳定性分析：

不稳定，例子：$[5,3,5,2 ]$，排序后变成$[2,3,5,5]$。

选择排序不稳定的原因是因为使用了swap操作，可以将swap操作换成insert操作，这样对于相同的值就能保持原序列。

```python
test_list = [2, 4, 1, 3, 10, 9, 5]
def select_sort(l):
    length = len(l)
    for i in range(length):
        min_idx = i
        min_value = l[i]
        for j in range(i,length):
            if l[j] < l[min_idx]:
                min_idx = j
        min_value = l[min_idx]
        # insert
        print("i:{}, min_idx:{}, min:{}, arr:{}".format(i,min_idx,min_value,l))
        for k in range(i,min_idx)[::-1]:
            l[k+1] = l[k]
        l[i] = min_value
        print("arr:{}".format(l))
	return l
select_sort(test_list) 
```

以上这种情况，仅在数据结构是链表的情况下才（不增加时间复杂度的条件下）成立

> Why Selection sort can be stable or unstable[https://stackoverflow.com/questions/20761396/why-selection-sort-can-be-stable-or-unstable]
>
> 通常情况下 - 你不正确。 选择排序不稳定。 这来自它的定义。 所以，你显然对一个自定义案例感到困惑。
>
> 它可以是稳定的 - 通常只有链表。 要做到这一点（经典的方式，O（1）内存） - 而不是交换，最小元素必须链接到未排序的部分，使整个算法稳定。 这就是“实现”的不同之处 - 显然，由于数据结构的具体情况，这种情况只能在这种情况下产生稳定性。 当选择排序不稳定时，它与常见情况无关

> 离开数据结构谈算法就是耍流氓！

从而引出问题，列表有几种实现方式？



#### 归并排序

```python
test_list = [2, 4, 1, 3, 10, 9, 5]
def merge_sort(l):
    if len(l) == 1:
        return l
    mid = len(l) // 2
    left = merge_sort(l[:mid])
    right = merge_sort(l[mid:])
    return merge(left,right)

# 方案1
def merge(left,right):
    if isinstance(left,int):
        left=[left]
    if isinstance(right,int):
        right=[right]
    length_l = len(left)
    length_r = len(right)
    length = length_l+length_r
    # 能不能创建固定长度的数组？
    ret_list =[0]*(length)
    i, j = 0, 0
    # while(i < length_l and j < length_r):
    for k in range(length):
        print("i:{},j:{},l:{}".format(i,j,ret_list))
        if left[i] < right[j]:
            ret_list[k] = left[i]
            i += 1
        else:
            ret_list[k] = right[j]
            j += 1
        if i == length_l:
            ret_list[k+1:] = right[j:]
            break
        elif j == length_r:
            ret_list[k+1:] = left[i:]
            break
    return ret_list

# 方案2
def merge(left,right):
    i, j = 0, 0
    ret_list = []
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            ret_list.append(left[i])
            i +=1
        else:
            ret_list.append(right[j])
            j +=1
        print("i:{}, j:{}, ret_list:{}".format(i, j, ret_list))
    ret_list += left[i:]
    ret_list += right[j:]
    return ret_list
    
```

时间复杂度推导，其中a=b=2。

时间复杂度：
$$
T(n)=2T(\frac{T}{2})+D(n)+C(n)
$$
$D(n)$ 为找中点需要的时间，$T(n)$ 是合并两个有序数组的时间

> ![1532056317848](pic\1532056317848.png)



稳定性：

只要考虑合并两个有序数组的过程，对于有序数组$arr_1,arr_2$，如果合并时，其插入的数据保持原数组中的顺序，则为稳定。



#### 堆排序

利用堆结构，对数据进行排序。主要包括建立堆，堆调整，首尾互换三个步骤。

堆调整指的是：对于一个二叉树结构，将最大\小值，放在根节点

```python
# 最大堆
test_list = [2, 4, 1, 3, 10, 9, 5]
def heap_sort(l):
    length = len(l)
    build_heap(l,length)
    for i in range(0, length)[::-1]:
        l[0], l[i] = l[i], l[0]
        adjust_heap(l,0,i)    
    return l
def build_heap(l,size):
    for i in range(size//2)[::-1]:
        adjust_heap(l,i,size)
def adjust_heap(l,i,size):
    left_idx = 2*i + 1
    right_idx = 2*i + 2
    max_idx = i
    if left_idx < size and l[left_idx] > l[max_idx]:
        max_idx = left_idx
    if right_idx < size and l[right_idx] > l[max_idx]:
        max_idx = right_idx
    if max_idx != i:
        l[max_idx], l[i] = l[i], l[max_idx]
        adjust_heap(l,max_idx,size)
heap_sort(test_list)   
test_list
```

```python
# 实例，找出一个list的top K
def heapfy(arr,i,size):
    l = 2*i + 1
    r = 2*i + 2
    largest = i
    if l<size and arr[l]>largest:
        largest = l
    if r<size and arr[r]>largest:
        largest = r
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapfy(arr,largest,size)

def top_k(arr,k):
    # build heap
    N = len(arr)
    top_k = []
    for i in range(N//2)[::-1]:
        heapfy(arr,i,N)
    # heapfy
    for i in range(k)[::-1]:
        arr[0], arr[i] = arr[i], arr[0]
        top_k.append(arr[i])
        heapfy(arr,0,i)
    return top_k

if __name__=='__main__':
    print(top_k([12,5,787,1,23],3))
```



##### 复杂度分析

建堆的复杂度计算(siftDown)
$$
S=\sum^{H}_{k=1}k \cdot 2^{H-k}=1\cdot2^{H-1}+2\cdot2^{H-2}+\dots+(H-1)\cdot2^1+H\cdot2^0  \\
\frac{1}{2}S=\sum^{H}_{k=1}k \cdot 2^{H-k-1}=1\cdot2^{H-2}+2\cdot2^{H-3}+\dots+(H-1)\cdot2^0 +H\cdot2^{-1}\\
\therefore S - \frac{1}{2}S=1\cdot2^{H-1}+(2-1)\cdot2^{H-2}+\dots+(H-(H-1))\cdot2^{0}-H\cdot2^{-1} \\
\therefore \frac{1}{2}S=1\cdot(2^H-1)-\frac{H}{2} \\
\because H \sim  \log_2n \\
\therefore S=2n-2-\log_2n \\
\therefore complexity = O(n)
$$

几个关键词

- siftDown
- siftUp



堆排序需要几个步骤：

1. buildHeap。即让数列具有堆性，这个过程需要用到heapify操作。heapify有两种方法，siftDown和siftUp，其中siftDown是从根节点开始向下移动，复杂度为$O(n)$，siftUp是从叶子节点向上移动，复杂度为$O(\log n)$。
2. 将最大值与最小值位置替换。这一步操作是通过删除最大值并将堆的最后一项与之替换来实现的，其复杂度为$O(\log n)$。

对于HeapSort的复杂度而言，步骤二的占主导地位（$O(n)+n*单次删除操作$），每次删除操作中，需要移动$h$次，对于同层的点而言，移动的次数是相同的，因此对于整个过程而言，复杂度为
$$
h * n / 2 +（h-1）* n / 4 + ... + 0 * 1 \sim n\log_2n-n \sim O(n\log_2n) \\
h\sim \log n
$$
稳定性：

不稳定



#### 快速排序

> It picks an element as pivot and partitions the given array around the picked pivot.There are many different versions of quickSort that pick pivot in different ways. 
>
> 1. Always pick first element as pivot.
> 2. Always pick last element as pivot (implemented below)
> 3. Pick a random element as pivot.
> 4. Pick median as pivot.
>
> The key process in quickSort is partition(). Target of partitions is, given an array and an element x of array as pivot, put x at its correct position in sorted array and put all smaller elements (smaller than x) before x, and put all greater elements (greater than x) after x. All this should be done in linear time. 

方法1

```python
test_list = [2, 4, 1, 3, 10, 9, 5]

def quick_sort(arr,start,end):
    if start < end:
        pi = _sort(arr,start,end)
        quick_sort(arr,start,pi-1)
        quick_sort(arr,pi+1,end)
def _sort(arr,start,end):
    print("arr: ",arr[start:end+1])
    if start >= end:
        return
    left = start 
    right = end
    pivot = arr[end]
    print("left:{}, right:{}".format(left,right))
    while(left < right):
        print("left:{}, right:{}".format(left,right))
        while(left < right and arr[left] < pivot):
            left += 1
        while(left < right and arr[right] >= pivot):
            right -= 1
        arr[left], arr[right] = arr[right], arr[left]
    arr[right], arr[end] = arr[end], arr[right]
    return right
    
quick_sort(test_list,0,len(test_list)-1)
```



方法2

```python
'''
i 是小大值的交界点，负责找 < privot 的数
'''

test_list = [2, 4, 1, 3, 10, 9, 5]

def quick_sort(arr,start,end):
    if start < end:
        pi = _sort(arr,start,end)
        quick_sort(arr,start,pi-1)
        quick_sort(arr,pi+1,end)

def _sort(arr,start,end):
    i = start -1
    pivot = arr[end]
    print("arr:{}".format(arr[start:end+1]))
    for j in range(start,end+1):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i+1], arr[end] = arr[end], arr[i+1]
    return i+1
quick_sort(test_list,0,len(test_list)-1)
```



方法3

```python
test_list = [2, 4, 1, 3, 10, 9, 5]
quick_sort = lambda array: array if len(array) <= 1 else quick_sort([item for item in array[1:] if item <= array[0]]) + [array[0]] + quick_sort([item for item in array[1:] if item > array[0]])
quick_sort
```



##### 复杂度分析

快排主要步骤是partition process和recursive。对于一般情况而言，快排的复杂度为：
$$
T(n)=T(k)+T(n-k-1)+O(n)
$$
其中k为小与pivot点元素的个数。快排的复杂度取决于partition strategy，分如下三种情况考虑。

1. 最好情况

   最好情况是每次均等划分$T(n)=T(k)+T(n-k-1)+O(n)$ ，$其中k=\frac{1}{2}$，可由主定理证明其复杂度为$O(n \log n)$。

2. 最坏情况

   最坏情况每次都是极端划分，比如每次pivot点都在最左或者最右侧，即$T(n)=T(0)+T(n-1)+O(n)$，每次子问题的规模减少1，其复杂度为$O(n^2)$。

3. 平均情况

   平均情况的性能与最好性能相近。



稳定性：

不稳定



几个问题：

1. 快排为什么快？
2. 快排有什么不足？
3. 如何优化快排？



### 4. 内排序（线性时间排序）

基于比较的排序，其时间复杂度的极限为$\Theta(n\log n)$，想要突破该极限需要考虑其他的排序方法。常见的线性排序方法有三个：

1. [Counting Sort](https://www.geeksforgeeks.org/counting-sort/)，计数排序
2. [Bucket Sort](https://www.geeksforgeeks.org/bucket-sort-2/)，桶排序
3. [Radix Sort](https://www.geeksforgeeks.org/radix-sort/)，基数排序

#### 1. 计数排序

**适用范围：**确定范围内的整数 ，比如char(0~256)

**Time Complexity:** O(n+k) where n is the number of elements in input array and k is the range of input. **Auxiliary Space:** O(n+k) 

原理：对每个元素进行计数，得到元素A前共有$i$个数， 则A应位于第$i$个位置（第$i-1$个索引处）

```python
# 对一串字符串进行排序，比如“showmethemoney”
def counting_sort(arr):
    cnt = [0 for i in range(256)]
    # ret = [0 for i in range(256)]
    ret = ["" for _ in arr]
    for char in arr:
        cnt[ord(char)] += 1
    for i in range(1,256):
        cnt[i] += cnt[i-1]
    for char in arr[::-1]:
        ret[cnt[ord(char)]-1] = char
    	cnt[ord(char)] -= 1
    return ret
        
test_list = "showmethemoney"
counting_sort(test_list)
```



#### 2. 桶排序

**适用范围**：数据集合范围不大。

假设条件：假设数据均匀的分为在一个区间内。更广的假设为，桶尺寸的平方和与总元素数成正比，即$E(n_i^2) \sim N$。

>  **Time Complexity:** If we assume that insertion in a bucket takes O(1) time then steps 1 and 2 of the above algorithm clearly take O(n) time. The O(1) is easily possible if we use a linked list to represent a bucket (In the following code, C++ vector is used for simplicity). Step 4 also takes O(n) time as there will be n items in all buckets. 
>
> The main step to analyze is step 3. *This step also takes O(n) time on average if all numbers are uniformly distributed* (please refer [CLRS book](http://www.flipkart.com/introduction-algorithms-3rd/p/itmdvd93bzvrnc7b?pid=9788120340077&affid=sandeepgfg)for more details) 

原理：对于有n个数据的数据集合，将数据放入N个桶中，每个桶为一个数值范围；对每个桶内的数据采用插入排序，最后按照桶编号依次输出数据。

思考：[Why do we use Insertion Sort in the Bucket Sort?](https://stackoverflow.com/questions/33405163/why-do-we-use-insertion-sort-in-the-bucket-sort)

```python
def bsort(A):
  """Returns A sorted. with A = {x : x such that 0 <= x < 1}."""
    buckets = [[] for x in range(10)]
    for i, x in enumerate(A):
        buckets[int(x*len(buckets))].append(x)
    out = []
    for buck in buckets:
        out += insert_sort(buck)
    return out
```



#### 3. 基数排序

原理：从最后一个对数据开始排序，然后对次高位开始排序，直到对最高位排序，完成排序。

```python
def radixSort(lista):
    RADIX = 10
    maxLength = False
    tmp = -1
    placement = 1
  
    while not maxLength:
        maxLength = True
        buckets = [list() for _ in range( RADIX )]
        
  
        for i in lista:
            tmp = i // placement
            buckets[tmp % RADIX].append( i )
            if maxLength and tmp > 0:
                maxLength = False

        a = 0
        for b in range( RADIX ):
            buck = buckets[b]
            for i in buck:
                lista[a] = i
                a += 1
        print(lista)
        print(buckets)
        placement *= RADIX
```



### 5. 外排序

基本思想如下：

> 外部排序指的是大文件的排序，即待排序的记录存储在外部存储器上，在排序过程中需进行多次的内、外存之间的交换。首先将打文件记录分成若干个子文件，然后读入内存中，并利用**内部排序**的方法进行排序；然后把排序好的有序子文件（称为：归并段）重新写入外存，再对这些归并段进行逐个归并，直到整个有序文件为止。      例子：假设有10 000个记录的文件需要排序，则先把这10 000个记录文件分成10个归并段（R1…~R10，每段1000个记录），然后逐个读入归并段进行内部排序（共10次），然后把排序好的归并段重新写入外存中，再进行两两归并，直到得到一个有序文件为止。 

主要举两个例：

>  [Example of Two-Way Sorting](http://faculty.simpson.edu/lydia.sinapova/www/cmsc250/LN250_Weiss/L17-ExternalSortEX1.htm) and [Example of multiway external sorting](http://faculty.simpson.edu/lydia.sinapova/www/cmsc250/LN250_Weiss/L17-ExternalSortEX2.htm)

思路如下：

对于N个数字，单次只能读取M个。则将N个数字分成$\lceil \frac{N}{M} \rceil$ 段，将每一段放入内存中排序，结果存放在$k$个外部文件中，然后对于$k$个外部文件的第$i$段，从左到右，逐每次取一个元素共计k个元素build heap, heapify之后deleteMin or deleteMax，然后塞入下一个元素，直到第$i$段被排序完毕，得到长度为$2M$的结果，存放在外部文件中。依次对所有段都进行同样的merge操作，然后合并所有段，合并次数为$\lceil \log_k(\frac{N}{M})\rceil$次，最后得到已排序的数组。
$$
[14,51,41|15,42,12|54,12,43] \\
[43,22,64,|21,12,23|49,42,36] \\
\vdots  \\
[42,56,21|67,45,25|35,13,75] \\
-------------\\
[14,21,22,41,42,43,51,56,64|...]
$$

## 3. 思考问题

Q : shell是如何对大文件进行排序的？

A : External R-Way merge sort，其实就是外排序中介绍的算法。



## 4. 常见问题与解法

### 1. 大量数据中找重复率最高的。

有10个G的数据，如果两条数据一样，则表示该两条数据重复了， 现在给你512M的内存，把这10G中重复次数最高的10条数据取出来。 

解法

找重复：

>1. 先排序， 10G数据分成40份，每份256M，排序，合并相同数据并加上计数器，写到临时文件chunk01~chunk20。 
>2. 对每一chunk, 读入内存，对每一条数据，再依次读入其后续个chunk, 合并相同数据的计数，后写入一个文件count。为了避免重复计数，在计数累加后需要将原来chunk的计数清零并回写文件。 以chunk01为例。假设chunk01中有数据A-8(数据A, 8次)，chunk02中有A-2，那么合并chunk02后 chunk01的内存中为A-10， chunk02中为A-0，这时把chunk02写回文件，然后读入chunk03继续处理处理。最后把chunk01中计数不为0的数据(chunk01里不会有计数为0的，但是后面的chunk会有)写入文件count. 
>3. 对count文件进行按重复次数排序。（分组，排序，然后每组选前10，再排序） 



> 个人觉得，分组统计，最后合并的方法是不可取的 因为有可能某个值A，在你每个分组中出现的次数都没有排进前10，但是将它在每个分组中的次数加起来，是能排进前10的。  
>
> 所以应该还是计数排序  其次是这10G数据时什么，是10G个BYTE，还是10G个字符串。
>
> 如果是字节，BYTE的范围0-255，也就是说这个问题就变成了，找0-255范围内，出现次数最多的10个数， 用int64[255]来计数，遍历一次，每个数值对应下标里面记录其出现的次数就可以了，用int64是因为DWORD表示不了10G。  
>
> 如果是字符串，或者是其他2进制数据块，就比较复杂，需要多次遍历 2进制数据块有多少个字节，就需要准备多少个int64[255]计数数组 假定每条记录，就是每个2进制数据块长10个字节，对于不足10字节的记录，不足的部分以0计算 需要的计数数组为`int64[10][255] ` 对于每条记录的第一个字节相同的，算为相同记录的情况下，得出表：
>
>  ```
> A*** 1000次 
> B*** 900次 
> C*** 900次 
> D*** 890次
>  ```
>
>  ...统计结果计入int64\[0\]\[255\]  然后对于1000次的`A*********，统计***`
>
> ```
> AE**** 50次 
> AA**** 50次 
> AD**** 49次 
> AC**** 47次 ...
> ```
>
> 统计结果计入`int64[1][255] `依此类推 
>
> ```
> AEDBBCADE* 10次 
> AEDBBCADB* 9次 
> AEDBBCADC* 9次 
> AEDBBCADA* 8次 ...
> ```
>
> 统计结果计入`int64[8][255] `最终
>
> ```
> AEDBBCADEA 3次 
> AEDBBCADEF 3次 
> AEDBBCADEC 2次 
> AEDBBCADEB 2次 ...
> ```
>
> 统计结果计入`int64[9][255] ` 将这个结果放入另一个int64[255] res，这个就是当前的最终结果  然后逐个向前递归 对于int64[8][255]中排行第二的“AEDBBCADB* 9次” 统计出前10，得到一个新的int64\[9\]\[255\]，将其与res中的结果比较，得出这20个中的前10，更新res  依此类推，得出最终结果 。



### 2. TopK

Q1：百度面试题

 搜索引擎会通过日志文件把用户每次检索使用的所有检索串都记录下来，每个查询串的长度为1-255字节。     假设目前有一千万个记录（这些查询串的重复度比较高，虽然总数是1千万，但如果除去重复后，不超过3百万个。一个查询串的重复度越高，说明查询它的用户越多，也就是越热门。），请你统计最热门的10个查询串，要求使用的内存不能超过1G。 

解法

思路：使用hashtable统计（复杂度为$\Theta(N)$），其key为$f(url)$，value为次数与url。然后排序，排序可以选择几种方法。1）维护一个K长的数组，其中放URL，然后逐个遍历所有URL，最终得到TopK。这个比较过程的复杂度为$\Theta(N*K)$。2）维护一个K个点组成的最小堆，遍历所有URL，执行deletaMin操作，最终得到TopK。这个比较过程的复杂度为$\Theta(N\log K)$



Q2：m个数组，每个数组都为n个整型数（递减排序），找出前k个最大的数（k < m*n），时间复杂度o(n)，空间复杂度o(1) 





# 2. 搜索

## 1. 二分法

使用迭代来写

```python
def binary_search(arr,tar):
    start = 0
    end = len(arr)-1
    return _search(arr,start,end,tar)
def _search(arr,start,end,tar):
    # 不可以start == end作为结束条件，因为可能arr=[8],tar=8，这时应该返回0，而不是-1
    if start>end:
        return -1
    mid = (start + end) // 2
    print("start:{}, end:{}, mid:{}, tar:{}, arr[mid]:{}".format(start,end,mid,tar,arr[mid]))
    if tar == arr[mid]:
        return mid
    elif tar > arr[mid]:
        return _search(arr,mid+1,end,tar)
    elif tar < arr[mid]:
        return _search(arr,start,mid-1,tar)
_search([1,2,5,8,12,13],0,5,8)
```



## 2. 哈希表

hash是一种思想，将大数映射到小范围的数中。简单的说就是一种将任意长度的消息压缩到某一固定长度的消息摘要的函数。 

hashing中，处理冲突的两种方法：

1. **Chaining** 
2. **Open Addressing** 

第一种方式，通过链表解决冲突问题。第二种方式，当地址冲突时，寻找下一个地址作为新地址，寻找的方式有三种：Linear Probing，Quadratic Probing，Double Hashing。

1. Linear Probing。当出现冲突时，找$(hash(x)+1) \% S$，如继续冲突，则 + 2.
2. Quadratic Probin。当出现冲突时，找$(hash(x)+1*1) \% S$，如继续冲突，则 + 2*2.
3. Double Hashing。当出现冲突时，找$(hash(x)+1*hash2(x)) \% S$，如继续冲突，则 + $2*hash2(x)$.

| S.No. | **Seperate Chaining**                                        | **Open Addressing**                                          |
| ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1.    | Chaining is Simpler to implement.                            | Open Addressing requires more computation.                   |
| 2.    | In chaining, Hash table never fills up, we can always add more elements to chain. | In open addressing, table may become full.                   |
| 3.    | Chaining is Less sensitive to the hash function or load factors. | Open addressing requires extra care for to avoid clustering and load factor. |
| 4.    | Chaining is mostly used when it is unknown how many and how frequently keys may be inserted or deleted. | Open addressing is used when the frequency and number of keys is known. |
| 5.    | Cache performance of chaining is not good as keys are stored using linked list. | Open addressing provides better cache performance as everything is stored in the same table. |
| 6.    | Wastage of Space (Some Parts of hash table in chaining are never used). | In Open addressing, a slot can be used even if an input doesn’t map to it. |
| 7.    | Chaining uses extra space for links.                         | No links in Open addressing                                  |



哈希表的搜索复杂度为$\Theta(1)$，因为通过给定的元素，能使用哈希函数直接算出其对应的索引位置。

## *. 一些思考

Q ：为什么数组中通过索引查找数据其复杂度为$\Theta(1)$？

> 一：一个数组中的元素在内存中的存放是连续的，一条链表中的元素在内存中的存放不一定是连续的。
> 二：链表的实现是在当前元素中划分出一块指针域指向下一个元素所在地址。
> 三：数组名这个变量中存放的是你这个数组中第一个元素的地址，链表的头结点中存放链表中第一个元素的地址。(不绝对，取决于你的写法)
> 四：计算机读写内存中的数据是建立在地址之上的，即根据给定的地址进行相应操作。
> 五：数组下标的实际意义是：相对于数组起始地址的地址偏移量。
>       有如下的等价关系：a[i] = *(a + i)

A：简而言之，数组的地址是连续的，通过索引能很快算出索引

 

 

 

 

 





# 3. Python数据结构

## 1. Python内部数据结构

原子性

> 注意：
>
> 　　　　　　　　>>> a=[1,2,3,4]                                                >>> a=1
>
> 　　　　　　　　>>> b=a                                                            >>> b=a
>
> 　　　　　　　　>>> a[0]=None                                 　　　　     >>> a=2
>
> 　　　　　　　　>>> b                                                                 >>> b
>
> 　　　　　　　　　　[None, 2, 3, 4]                                                1
>
> 　　　　　　　　这里就出现了一个很奇怪的现象了，左边的b跟着a变而右边的没有，其实这就是因为int为原子性数据结构带来的。可以认为原子性数据所在内存地址是固定的，这样改变了a变量的值就是改变了a变量指向的地址。而非原子性改变a就是改变了a所指内存地址所存储的东西而没有改变a指向的地址。这样就能够解释通了

> ​      　　 Python内部数据从某种形式上可以分为两种：
>
> 　　　　　　其一是原子性数据类型：int,float,str
>
> 　　　　　　其余的是非原子性（容器类型）的(按有序性分)：
>
> 　　　　　　　　有序的：list, tuple
>
> 　　　　　　　　无序的：set, dict

> 原子类型，可以简单理解为数据不可拆分的类型，但不完成正确，类似C的简单类型/内置类型 可以用组成来理解：数值(所有的数值类型), 字符串(全部是文字)  

## 2. 各数据类型操作的复杂度

![img](pic\list.png) 

![img](pic\set.png) 

![img](pic\dict.png) 



# 4. 图论



# 5

# . 推荐算法

