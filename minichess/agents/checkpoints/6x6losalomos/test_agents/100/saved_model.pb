ђу
╩«
D
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Џ
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(ѕ

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Й
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕ
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.6.02unknown8лђ
ю
res_block_output_base/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameres_block_output_base/kernel
Ћ
0res_block_output_base/kernel/Read/ReadVariableOpReadVariableOpres_block_output_base/kernel*&
_output_shapes
:@*
dtype0
ї
res_block_output_base/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameres_block_output_base/bias
Ё
.res_block_output_base/bias/Read/ReadVariableOpReadVariableOpres_block_output_base/bias*
_output_shapes
:@*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:@@*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:@*
dtype0
ѓ
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0
ѓ
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:@*
dtype0
ѓ
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:@*
dtype0
ѓ
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:@*
dtype0
ѓ
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:@*
dtype0
ѓ
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:@*
dtype0
є
value_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_namevalue_conv/kernel

%value_conv/kernel/Read/ReadVariableOpReadVariableOpvalue_conv/kernel*&
_output_shapes
:@*
dtype0
v
value_conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namevalue_conv/bias
o
#value_conv/bias/Read/ReadVariableOpReadVariableOpvalue_conv/bias*
_output_shapes
:*
dtype0
ѓ
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?* 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
:@?*
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
:?*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	└ *
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	└ *
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
: *
dtype0
ѓ
value_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *$
shared_namevalue_output/kernel
{
'value_output/kernel/Read/ReadVariableOpReadVariableOpvalue_output/kernel*
_output_shapes

: *
dtype0
z
value_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namevalue_output/bias
s
%value_output/bias/Read/ReadVariableOpReadVariableOpvalue_output/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
ф
#Adam/res_block_output_base/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/res_block_output_base/kernel/m
Б
7Adam/res_block_output_base/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/res_block_output_base/kernel/m*&
_output_shapes
:@*
dtype0
џ
!Adam/res_block_output_base/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/res_block_output_base/bias/m
Њ
5Adam/res_block_output_base/bias/m/Read/ReadVariableOpReadVariableOp!Adam/res_block_output_base/bias/m*
_output_shapes
:@*
dtype0
ї
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*%
shared_nameAdam/conv2d/kernel/m
Ё
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:@@*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:@*
dtype0
љ
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_1/kernel/m
Ѕ
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:@@*
dtype0
ђ
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:@*
dtype0
љ
Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_2/kernel/m
Ѕ
*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
:@@*
dtype0
ђ
Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes
:@*
dtype0
љ
Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_3/kernel/m
Ѕ
*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*&
_output_shapes
:@@*
dtype0
ђ
Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_3/bias/m
y
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
_output_shapes
:@*
dtype0
љ
Adam/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_4/kernel/m
Ѕ
*Adam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/m*&
_output_shapes
:@@*
dtype0
ђ
Adam/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_4/bias/m
y
(Adam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/m*
_output_shapes
:@*
dtype0
љ
Adam/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_5/kernel/m
Ѕ
*Adam/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/m*&
_output_shapes
:@@*
dtype0
ђ
Adam/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_5/bias/m
y
(Adam/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/m*
_output_shapes
:@*
dtype0
љ
Adam/conv2d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_6/kernel/m
Ѕ
*Adam/conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/m*&
_output_shapes
:@@*
dtype0
ђ
Adam/conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_6/bias/m
y
(Adam/conv2d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/m*
_output_shapes
:@*
dtype0
ћ
Adam/value_conv/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/value_conv/kernel/m
Ї
,Adam/value_conv/kernel/m/Read/ReadVariableOpReadVariableOpAdam/value_conv/kernel/m*&
_output_shapes
:@*
dtype0
ё
Adam/value_conv/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/value_conv/bias/m
}
*Adam/value_conv/bias/m/Read/ReadVariableOpReadVariableOpAdam/value_conv/bias/m*
_output_shapes
:*
dtype0
љ
Adam/conv2d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*'
shared_nameAdam/conv2d_7/kernel/m
Ѕ
*Adam/conv2d_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/m*&
_output_shapes
:@?*
dtype0
ђ
Adam/conv2d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv2d_7/bias/m
y
(Adam/conv2d_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/m*
_output_shapes
:?*
dtype0
Ѓ
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	└ *$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	└ *
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
: *
dtype0
љ
Adam/value_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *+
shared_nameAdam/value_output/kernel/m
Ѕ
.Adam/value_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/value_output/kernel/m*
_output_shapes

: *
dtype0
ѕ
Adam/value_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/value_output/bias/m
Ђ
,Adam/value_output/bias/m/Read/ReadVariableOpReadVariableOpAdam/value_output/bias/m*
_output_shapes
:*
dtype0
ф
#Adam/res_block_output_base/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/res_block_output_base/kernel/v
Б
7Adam/res_block_output_base/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/res_block_output_base/kernel/v*&
_output_shapes
:@*
dtype0
џ
!Adam/res_block_output_base/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/res_block_output_base/bias/v
Њ
5Adam/res_block_output_base/bias/v/Read/ReadVariableOpReadVariableOp!Adam/res_block_output_base/bias/v*
_output_shapes
:@*
dtype0
ї
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*%
shared_nameAdam/conv2d/kernel/v
Ё
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:@@*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:@*
dtype0
љ
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_1/kernel/v
Ѕ
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:@@*
dtype0
ђ
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:@*
dtype0
љ
Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_2/kernel/v
Ѕ
*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
:@@*
dtype0
ђ
Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_2/bias/v
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
:@*
dtype0
љ
Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_3/kernel/v
Ѕ
*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*&
_output_shapes
:@@*
dtype0
ђ
Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_3/bias/v
y
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
_output_shapes
:@*
dtype0
љ
Adam/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_4/kernel/v
Ѕ
*Adam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/v*&
_output_shapes
:@@*
dtype0
ђ
Adam/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_4/bias/v
y
(Adam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/v*
_output_shapes
:@*
dtype0
љ
Adam/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_5/kernel/v
Ѕ
*Adam/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/v*&
_output_shapes
:@@*
dtype0
ђ
Adam/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_5/bias/v
y
(Adam/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/v*
_output_shapes
:@*
dtype0
љ
Adam/conv2d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_6/kernel/v
Ѕ
*Adam/conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/v*&
_output_shapes
:@@*
dtype0
ђ
Adam/conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_6/bias/v
y
(Adam/conv2d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/v*
_output_shapes
:@*
dtype0
ћ
Adam/value_conv/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/value_conv/kernel/v
Ї
,Adam/value_conv/kernel/v/Read/ReadVariableOpReadVariableOpAdam/value_conv/kernel/v*&
_output_shapes
:@*
dtype0
ё
Adam/value_conv/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/value_conv/bias/v
}
*Adam/value_conv/bias/v/Read/ReadVariableOpReadVariableOpAdam/value_conv/bias/v*
_output_shapes
:*
dtype0
љ
Adam/conv2d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*'
shared_nameAdam/conv2d_7/kernel/v
Ѕ
*Adam/conv2d_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/v*&
_output_shapes
:@?*
dtype0
ђ
Adam/conv2d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv2d_7/bias/v
y
(Adam/conv2d_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/v*
_output_shapes
:?*
dtype0
Ѓ
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	└ *$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	└ *
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
: *
dtype0
љ
Adam/value_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *+
shared_nameAdam/value_output/kernel/v
Ѕ
.Adam/value_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/value_output/kernel/v*
_output_shapes

: *
dtype0
ѕ
Adam/value_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/value_output/bias/v
Ђ
,Adam/value_output/bias/v/Read/ReadVariableOpReadVariableOpAdam/value_output/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
їј
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*кЇ
value╗ЇBиЇ B»Ї
▀
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer-12
layer-13
layer_with_weights-7
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
layer-17
layer-18
layer_with_weights-10
layer-19
layer-20
layer_with_weights-11
layer-21
	optimizer
loss
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
h

kernel
bias
 trainable_variables
!regularization_losses
"	variables
#	keras_api
h

$kernel
%bias
&trainable_variables
'regularization_losses
(	variables
)	keras_api
h

*kernel
+bias
,trainable_variables
-regularization_losses
.	variables
/	keras_api
R
0trainable_variables
1regularization_losses
2	variables
3	keras_api
R
4trainable_variables
5regularization_losses
6	variables
7	keras_api
h

8kernel
9bias
:trainable_variables
;regularization_losses
<	variables
=	keras_api
h

>kernel
?bias
@trainable_variables
Aregularization_losses
B	variables
C	keras_api
R
Dtrainable_variables
Eregularization_losses
F	variables
G	keras_api
R
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
h

Lkernel
Mbias
Ntrainable_variables
Oregularization_losses
P	variables
Q	keras_api
h

Rkernel
Sbias
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
R
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
R
\trainable_variables
]regularization_losses
^	variables
_	keras_api
h

`kernel
abias
btrainable_variables
cregularization_losses
d	variables
e	keras_api
h

fkernel
gbias
htrainable_variables
iregularization_losses
j	variables
k	keras_api
h

lkernel
mbias
ntrainable_variables
oregularization_losses
p	variables
q	keras_api
R
rtrainable_variables
sregularization_losses
t	variables
u	keras_api
R
vtrainable_variables
wregularization_losses
x	variables
y	keras_api
h

zkernel
{bias
|trainable_variables
}regularization_losses
~	variables
	keras_api
V
ђtrainable_variables
Ђregularization_losses
ѓ	variables
Ѓ	keras_api
n
ёkernel
	Ёbias
єtrainable_variables
Єregularization_losses
ѕ	variables
Ѕ	keras_api
Е
	іiter
Іbeta_1
їbeta_2

Їdecay
јlearning_ratemїmЇ$mј%mЈ*mљ+mЉ8mњ9mЊ>mћ?mЋLmќMmЌRmўSmЎ`mџamЏfmюgmЮlmъmmЪzmа{mА	ёmб	ЁmБvцvЦ$vд%vД*vе+vЕ8vф9vФ>vг?vГLv«Mv»Rv░Sv▒`v▓av│fv┤gvхlvХmvиzvИ{v╣	ёv║	Ёv╗
 
И
0
1
$2
%3
*4
+5
86
97
>8
?9
L10
M11
R12
S13
`14
a15
f16
g17
l18
m19
z20
{21
ё22
Ё23
 
И
0
1
$2
%3
*4
+5
86
97
>8
?9
L10
M11
R12
S13
`14
a15
f16
g17
l18
m19
z20
{21
ё22
Ё23
▓
Јmetrics
љlayers
Љlayer_metrics
trainable_variables
regularization_losses
	variables
 њlayer_regularization_losses
Њnon_trainable_variables
 
hf
VARIABLE_VALUEres_block_output_base/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEres_block_output_base/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
▓
ћmetrics
Ћlayers
ќlayer_metrics
 trainable_variables
!regularization_losses
"	variables
 Ќlayer_regularization_losses
ўnon_trainable_variables
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1
 

$0
%1
▓
Ўmetrics
џlayers
Џlayer_metrics
&trainable_variables
'regularization_losses
(	variables
 юlayer_regularization_losses
Юnon_trainable_variables
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1
 

*0
+1
▓
ъmetrics
Ъlayers
аlayer_metrics
,trainable_variables
-regularization_losses
.	variables
 Аlayer_regularization_losses
бnon_trainable_variables
 
 
 
▓
Бmetrics
цlayers
Цlayer_metrics
0trainable_variables
1regularization_losses
2	variables
 дlayer_regularization_losses
Дnon_trainable_variables
 
 
 
▓
еmetrics
Еlayers
фlayer_metrics
4trainable_variables
5regularization_losses
6	variables
 Фlayer_regularization_losses
гnon_trainable_variables
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91
 

80
91
▓
Гmetrics
«layers
»layer_metrics
:trainable_variables
;regularization_losses
<	variables
 ░layer_regularization_losses
▒non_trainable_variables
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

>0
?1
 

>0
?1
▓
▓metrics
│layers
┤layer_metrics
@trainable_variables
Aregularization_losses
B	variables
 хlayer_regularization_losses
Хnon_trainable_variables
 
 
 
▓
иmetrics
Иlayers
╣layer_metrics
Dtrainable_variables
Eregularization_losses
F	variables
 ║layer_regularization_losses
╗non_trainable_variables
 
 
 
▓
╝metrics
йlayers
Йlayer_metrics
Htrainable_variables
Iregularization_losses
J	variables
 ┐layer_regularization_losses
└non_trainable_variables
[Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

L0
M1
 

L0
M1
▓
┴metrics
┬layers
├layer_metrics
Ntrainable_variables
Oregularization_losses
P	variables
 ─layer_regularization_losses
┼non_trainable_variables
[Y
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

R0
S1
 

R0
S1
▓
кmetrics
Кlayers
╚layer_metrics
Ttrainable_variables
Uregularization_losses
V	variables
 ╔layer_regularization_losses
╩non_trainable_variables
 
 
 
▓
╦metrics
╠layers
═layer_metrics
Xtrainable_variables
Yregularization_losses
Z	variables
 ╬layer_regularization_losses
¤non_trainable_variables
 
 
 
▓
лmetrics
Лlayers
мlayer_metrics
\trainable_variables
]regularization_losses
^	variables
 Мlayer_regularization_losses
нnon_trainable_variables
[Y
VARIABLE_VALUEconv2d_6/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_6/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

`0
a1
 

`0
a1
▓
Нmetrics
оlayers
Оlayer_metrics
btrainable_variables
cregularization_losses
d	variables
 пlayer_regularization_losses
┘non_trainable_variables
][
VARIABLE_VALUEvalue_conv/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEvalue_conv/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

f0
g1
 

f0
g1
▓
┌metrics
█layers
▄layer_metrics
htrainable_variables
iregularization_losses
j	variables
 Пlayer_regularization_losses
яnon_trainable_variables
[Y
VARIABLE_VALUEconv2d_7/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_7/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

l0
m1
 

l0
m1
▓
▀metrics
Яlayers
рlayer_metrics
ntrainable_variables
oregularization_losses
p	variables
 Рlayer_regularization_losses
сnon_trainable_variables
 
 
 
▓
Сmetrics
тlayers
Тlayer_metrics
rtrainable_variables
sregularization_losses
t	variables
 уlayer_regularization_losses
Уnon_trainable_variables
 
 
 
▓
жmetrics
Жlayers
вlayer_metrics
vtrainable_variables
wregularization_losses
x	variables
 Вlayer_regularization_losses
ьnon_trainable_variables
YW
VARIABLE_VALUEdense/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
dense/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

z0
{1
 

z0
{1
▓
Ьmetrics
№layers
­layer_metrics
|trainable_variables
}regularization_losses
~	variables
 ыlayer_regularization_losses
Ыnon_trainable_variables
 
 
 
х
зmetrics
Зlayers
шlayer_metrics
ђtrainable_variables
Ђregularization_losses
ѓ	variables
 Шlayer_regularization_losses
эnon_trainable_variables
`^
VARIABLE_VALUEvalue_output/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEvalue_output/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

ё0
Ё1
 

ё0
Ё1
х
Эmetrics
щlayers
Щlayer_metrics
єtrainable_variables
Єregularization_losses
ѕ	variables
 чlayer_regularization_losses
Чnon_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

§0
■1
 2
д
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

ђtotal

Ђcount
ѓ	variables
Ѓ	keras_api
8

ёtotal

Ёcount
є	variables
Є	keras_api
8

ѕtotal

Ѕcount
і	variables
І	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

ђ0
Ђ1

ѓ	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE

ё0
Ё1

є	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE

ѕ0
Ѕ1

і	variables
їЅ
VARIABLE_VALUE#Adam/res_block_output_base/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ѕЁ
VARIABLE_VALUE!Adam/res_block_output_base/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_5/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_5/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_6/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_6/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUEAdam/value_conv/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/value_conv/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_7/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_7/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ёЂ
VARIABLE_VALUEAdam/value_output/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/value_output/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
їЅ
VARIABLE_VALUE#Adam/res_block_output_base/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ѕЁ
VARIABLE_VALUE!Adam/res_block_output_base/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_5/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_5/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_6/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_6/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUEAdam/value_conv/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/value_conv/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_7/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_7/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ёЂ
VARIABLE_VALUEAdam/value_output/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/value_output/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
і
serving_default_input_1Placeholder*/
_output_shapes
:         *
dtype0*$
shape:         
Ў
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1res_block_output_base/kernelres_block_output_base/biasconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasvalue_conv/kernelvalue_conv/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasdense/kernel
dense/biasvalue_output/kernelvalue_output/bias*$
Tin
2*
Tout
2*
_collective_manager_ids
 *;
_output_shapes)
':         ▄:         *:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *.
f)R'
%__inference_signature_wrapper_1592566
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
І
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename0res_block_output_base/kernel/Read/ReadVariableOp.res_block_output_base/bias/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp%value_conv/kernel/Read/ReadVariableOp#value_conv/bias/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp'value_output/kernel/Read/ReadVariableOp%value_output/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp7Adam/res_block_output_base/kernel/m/Read/ReadVariableOp5Adam/res_block_output_base/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp*Adam/conv2d_4/kernel/m/Read/ReadVariableOp(Adam/conv2d_4/bias/m/Read/ReadVariableOp*Adam/conv2d_5/kernel/m/Read/ReadVariableOp(Adam/conv2d_5/bias/m/Read/ReadVariableOp*Adam/conv2d_6/kernel/m/Read/ReadVariableOp(Adam/conv2d_6/bias/m/Read/ReadVariableOp,Adam/value_conv/kernel/m/Read/ReadVariableOp*Adam/value_conv/bias/m/Read/ReadVariableOp*Adam/conv2d_7/kernel/m/Read/ReadVariableOp(Adam/conv2d_7/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp.Adam/value_output/kernel/m/Read/ReadVariableOp,Adam/value_output/bias/m/Read/ReadVariableOp7Adam/res_block_output_base/kernel/v/Read/ReadVariableOp5Adam/res_block_output_base/bias/v/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp*Adam/conv2d_4/kernel/v/Read/ReadVariableOp(Adam/conv2d_4/bias/v/Read/ReadVariableOp*Adam/conv2d_5/kernel/v/Read/ReadVariableOp(Adam/conv2d_5/bias/v/Read/ReadVariableOp*Adam/conv2d_6/kernel/v/Read/ReadVariableOp(Adam/conv2d_6/bias/v/Read/ReadVariableOp,Adam/value_conv/kernel/v/Read/ReadVariableOp*Adam/value_conv/bias/v/Read/ReadVariableOp*Adam/conv2d_7/kernel/v/Read/ReadVariableOp(Adam/conv2d_7/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp.Adam/value_output/kernel/v/Read/ReadVariableOp,Adam/value_output/bias/v/Read/ReadVariableOpConst*`
TinY
W2U	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *)
f$R"
 __inference__traced_save_1593484
і
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameres_block_output_base/kernelres_block_output_base/biasconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasvalue_conv/kernelvalue_conv/biasconv2d_7/kernelconv2d_7/biasdense/kernel
dense/biasvalue_output/kernelvalue_output/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2#Adam/res_block_output_base/kernel/m!Adam/res_block_output_base/bias/mAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/conv2d_3/kernel/mAdam/conv2d_3/bias/mAdam/conv2d_4/kernel/mAdam/conv2d_4/bias/mAdam/conv2d_5/kernel/mAdam/conv2d_5/bias/mAdam/conv2d_6/kernel/mAdam/conv2d_6/bias/mAdam/value_conv/kernel/mAdam/value_conv/bias/mAdam/conv2d_7/kernel/mAdam/conv2d_7/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/value_output/kernel/mAdam/value_output/bias/m#Adam/res_block_output_base/kernel/v!Adam/res_block_output_base/bias/vAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/vAdam/conv2d_4/kernel/vAdam/conv2d_4/bias/vAdam/conv2d_5/kernel/vAdam/conv2d_5/bias/vAdam/conv2d_6/kernel/vAdam/conv2d_6/bias/vAdam/value_conv/kernel/vAdam/value_conv/bias/vAdam/conv2d_7/kernel/vAdam/conv2d_7/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/value_output/kernel/vAdam/value_output/bias/v*_
TinX
V2T*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *,
f'R%
#__inference__traced_restore_1593743дЌ
ђ
l
@__inference_add_layer_call_and_return_conditional_losses_1592946
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:         @2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         @:         @:Y U
/
_output_shapes
:         @
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         @
"
_user_specified_name
inputs/1
э
P
4__inference_res_block_output_0_layer_call_fn_1592951

inputs
identity┌
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *X
fSRQ
O__inference_res_block_output_0_layer_call_and_return_conditional_losses_15916772
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Л
K
/__inference_policy_output_layer_call_fn_1593186

inputs
identity╬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ▄* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__inference_policy_output_layer_call_and_return_conditional_losses_15918822
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ▄2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ▄:P L
(
_output_shapes
:         ▄
 
_user_specified_nameinputs
ѓ
n
B__inference_add_2_layer_call_and_return_conditional_losses_1593070
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:         @2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         @:         @:Y U
/
_output_shapes
:         @
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         @
"
_user_specified_name
inputs/1
ш
Ћ
'__inference_dense_layer_call_fn_1593170

inputs
unknown:	└ 
	unknown_0: 
identityѕбStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_15918462
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         └: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
э
P
4__inference_res_block_output_1_layer_call_fn_1593013

inputs
identity┌
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *X
fSRQ
O__inference_res_block_output_1_layer_call_and_return_conditional_losses_15917262
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Ь
Q
%__inference_add_layer_call_fn_1592940
inputs_0
inputs_1
identityп
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_15916702
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         @:         @:Y U
/
_output_shapes
:         @
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         @
"
_user_specified_name
inputs/1
Ы
S
'__inference_add_1_layer_call_fn_1593002
inputs_0
inputs_1
identity┌
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_15917192
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         @:         @:Y U
/
_output_shapes
:         @
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         @
"
_user_specified_name
inputs/1
Ц
А
,__inference_value_conv_layer_call_fn_1593109

inputs!
unknown:@
	unknown_0:
identityѕбStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *P
fKRI
G__inference_value_conv_layer_call_and_return_conditional_losses_15917882
StatefulPartitionedCallЃ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
ь
ђ
G__inference_value_conv_layer_call_and_return_conditional_losses_1593120

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Яb
┴
B__inference_model_layer_call_and_return_conditional_losses_1592503
input_17
res_block_output_base_1592432:@+
res_block_output_base_1592434:@(
conv2d_1592437:@@
conv2d_1592439:@*
conv2d_1_1592442:@@
conv2d_1_1592444:@*
conv2d_2_1592449:@@
conv2d_2_1592451:@*
conv2d_3_1592454:@@
conv2d_3_1592456:@*
conv2d_4_1592461:@@
conv2d_4_1592463:@*
conv2d_5_1592466:@@
conv2d_5_1592468:@,
value_conv_1592473:@ 
value_conv_1592475:*
conv2d_6_1592478:@@
conv2d_6_1592480:@*
conv2d_7_1592484:@?
conv2d_7_1592486:? 
dense_1592489:	└ 
dense_1592491: &
value_output_1592495: "
value_output_1592497:
identity

identity_1ѕбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallб conv2d_2/StatefulPartitionedCallб conv2d_3/StatefulPartitionedCallб conv2d_4/StatefulPartitionedCallб conv2d_5/StatefulPartitionedCallб conv2d_6/StatefulPartitionedCallб conv2d_7/StatefulPartitionedCallбdense/StatefulPartitionedCallб-res_block_output_base/StatefulPartitionedCallб"value_conv/StatefulPartitionedCallб$value_output/StatefulPartitionedCallТ
-res_block_output_base/StatefulPartitionedCallStatefulPartitionedCallinput_1res_block_output_base_1592432res_block_output_base_1592434*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *[
fVRT
R__inference_res_block_output_base_layer_call_and_return_conditional_losses_15916242/
-res_block_output_base/StatefulPartitionedCall╩
conv2d/StatefulPartitionedCallStatefulPartitionedCall6res_block_output_base/StatefulPartitionedCall:output:0conv2d_1592437conv2d_1592439*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_15916412 
conv2d/StatefulPartitionedCall┼
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_1592442conv2d_1_1592444*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_15916582"
 conv2d_1/StatefulPartitionedCall»
add/PartitionedCallPartitionedCall6res_block_output_base/StatefulPartitionedCall:output:0)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_15916702
add/PartitionedCallќ
"res_block_output_0/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *X
fSRQ
O__inference_res_block_output_0_layer_call_and_return_conditional_losses_15916772$
"res_block_output_0/PartitionedCall╔
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall+res_block_output_0/PartitionedCall:output:0conv2d_2_1592449conv2d_2_1592451*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_15916902"
 conv2d_2/StatefulPartitionedCallК
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_1592454conv2d_3_1592456*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_15917072"
 conv2d_3/StatefulPartitionedCallф
add_1/PartitionedCallPartitionedCall+res_block_output_0/PartitionedCall:output:0)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_15917192
add_1/PartitionedCallў
"res_block_output_1/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *X
fSRQ
O__inference_res_block_output_1_layer_call_and_return_conditional_losses_15917262$
"res_block_output_1/PartitionedCall╔
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall+res_block_output_1/PartitionedCall:output:0conv2d_4_1592461conv2d_4_1592463*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_15917392"
 conv2d_4/StatefulPartitionedCallК
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_1592466conv2d_5_1592468*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_15917562"
 conv2d_5/StatefulPartitionedCallф
add_2/PartitionedCallPartitionedCall+res_block_output_1/PartitionedCall:output:0)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_add_2_layer_call_and_return_conditional_losses_15917682
add_2/PartitionedCallў
"res_block_output_2/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *X
fSRQ
O__inference_res_block_output_2_layer_call_and_return_conditional_losses_15917752$
"res_block_output_2/PartitionedCallМ
"value_conv/StatefulPartitionedCallStatefulPartitionedCall+res_block_output_2/PartitionedCall:output:0value_conv_1592473value_conv_1592475*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *P
fKRI
G__inference_value_conv_layer_call_and_return_conditional_losses_15917882$
"value_conv/StatefulPartitionedCall╔
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall+res_block_output_2/PartitionedCall:output:0conv2d_6_1592478conv2d_6_1592480*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_15918052"
 conv2d_6/StatefulPartitionedCallЃ
flatten_1/PartitionedCallPartitionedCall+value_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_15918172
flatten_1/PartitionedCallК
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_1592484conv2d_7_1592486*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_15918292"
 conv2d_7/StatefulPartitionedCallЕ
dense/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1592489dense_1592491*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_15918462
dense/StatefulPartitionedCallч
flatten/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ▄* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_15918582
flatten/PartitionedCallл
$value_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0value_output_1592495value_output_1592497*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *R
fMRK
I__inference_value_output_layer_call_and_return_conditional_losses_15918712&
$value_output/StatefulPartitionedCallё
policy_output/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ▄* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__inference_policy_output_layer_call_and_return_conditional_losses_15918822
policy_output/PartitionedCallѓ
IdentityIdentity&policy_output/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ▄2

Identityї

Identity_1Identity-value_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity_1ђ
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall^dense/StatefulPartitionedCall.^res_block_output_base/StatefulPartitionedCall#^value_conv/StatefulPartitionedCall%^value_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:         : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2^
-res_block_output_base/StatefulPartitionedCall-res_block_output_base/StatefulPartitionedCall2H
"value_conv/StatefulPartitionedCall"value_conv/StatefulPartitionedCall2L
$value_output/StatefulPartitionedCall$value_output/StatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_1
яс
╠4
#__inference__traced_restore_1593743
file_prefixG
-assignvariableop_res_block_output_base_kernel:@;
-assignvariableop_1_res_block_output_base_bias:@:
 assignvariableop_2_conv2d_kernel:@@,
assignvariableop_3_conv2d_bias:@<
"assignvariableop_4_conv2d_1_kernel:@@.
 assignvariableop_5_conv2d_1_bias:@<
"assignvariableop_6_conv2d_2_kernel:@@.
 assignvariableop_7_conv2d_2_bias:@<
"assignvariableop_8_conv2d_3_kernel:@@.
 assignvariableop_9_conv2d_3_bias:@=
#assignvariableop_10_conv2d_4_kernel:@@/
!assignvariableop_11_conv2d_4_bias:@=
#assignvariableop_12_conv2d_5_kernel:@@/
!assignvariableop_13_conv2d_5_bias:@=
#assignvariableop_14_conv2d_6_kernel:@@/
!assignvariableop_15_conv2d_6_bias:@?
%assignvariableop_16_value_conv_kernel:@1
#assignvariableop_17_value_conv_bias:=
#assignvariableop_18_conv2d_7_kernel:@?/
!assignvariableop_19_conv2d_7_bias:?3
 assignvariableop_20_dense_kernel:	└ ,
assignvariableop_21_dense_bias: 9
'assignvariableop_22_value_output_kernel: 3
%assignvariableop_23_value_output_bias:'
assignvariableop_24_adam_iter:	 )
assignvariableop_25_adam_beta_1: )
assignvariableop_26_adam_beta_2: (
assignvariableop_27_adam_decay: 0
&assignvariableop_28_adam_learning_rate: #
assignvariableop_29_total: #
assignvariableop_30_count: %
assignvariableop_31_total_1: %
assignvariableop_32_count_1: %
assignvariableop_33_total_2: %
assignvariableop_34_count_2: Q
7assignvariableop_35_adam_res_block_output_base_kernel_m:@C
5assignvariableop_36_adam_res_block_output_base_bias_m:@B
(assignvariableop_37_adam_conv2d_kernel_m:@@4
&assignvariableop_38_adam_conv2d_bias_m:@D
*assignvariableop_39_adam_conv2d_1_kernel_m:@@6
(assignvariableop_40_adam_conv2d_1_bias_m:@D
*assignvariableop_41_adam_conv2d_2_kernel_m:@@6
(assignvariableop_42_adam_conv2d_2_bias_m:@D
*assignvariableop_43_adam_conv2d_3_kernel_m:@@6
(assignvariableop_44_adam_conv2d_3_bias_m:@D
*assignvariableop_45_adam_conv2d_4_kernel_m:@@6
(assignvariableop_46_adam_conv2d_4_bias_m:@D
*assignvariableop_47_adam_conv2d_5_kernel_m:@@6
(assignvariableop_48_adam_conv2d_5_bias_m:@D
*assignvariableop_49_adam_conv2d_6_kernel_m:@@6
(assignvariableop_50_adam_conv2d_6_bias_m:@F
,assignvariableop_51_adam_value_conv_kernel_m:@8
*assignvariableop_52_adam_value_conv_bias_m:D
*assignvariableop_53_adam_conv2d_7_kernel_m:@?6
(assignvariableop_54_adam_conv2d_7_bias_m:?:
'assignvariableop_55_adam_dense_kernel_m:	└ 3
%assignvariableop_56_adam_dense_bias_m: @
.assignvariableop_57_adam_value_output_kernel_m: :
,assignvariableop_58_adam_value_output_bias_m:Q
7assignvariableop_59_adam_res_block_output_base_kernel_v:@C
5assignvariableop_60_adam_res_block_output_base_bias_v:@B
(assignvariableop_61_adam_conv2d_kernel_v:@@4
&assignvariableop_62_adam_conv2d_bias_v:@D
*assignvariableop_63_adam_conv2d_1_kernel_v:@@6
(assignvariableop_64_adam_conv2d_1_bias_v:@D
*assignvariableop_65_adam_conv2d_2_kernel_v:@@6
(assignvariableop_66_adam_conv2d_2_bias_v:@D
*assignvariableop_67_adam_conv2d_3_kernel_v:@@6
(assignvariableop_68_adam_conv2d_3_bias_v:@D
*assignvariableop_69_adam_conv2d_4_kernel_v:@@6
(assignvariableop_70_adam_conv2d_4_bias_v:@D
*assignvariableop_71_adam_conv2d_5_kernel_v:@@6
(assignvariableop_72_adam_conv2d_5_bias_v:@D
*assignvariableop_73_adam_conv2d_6_kernel_v:@@6
(assignvariableop_74_adam_conv2d_6_bias_v:@F
,assignvariableop_75_adam_value_conv_kernel_v:@8
*assignvariableop_76_adam_value_conv_bias_v:D
*assignvariableop_77_adam_conv2d_7_kernel_v:@?6
(assignvariableop_78_adam_conv2d_7_bias_v:?:
'assignvariableop_79_adam_dense_kernel_v:	└ 3
%assignvariableop_80_adam_dense_bias_v: @
.assignvariableop_81_adam_value_output_kernel_v: :
,assignvariableop_82_adam_value_output_bias_v:
identity_84ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_45бAssignVariableOp_46бAssignVariableOp_47бAssignVariableOp_48бAssignVariableOp_49бAssignVariableOp_5бAssignVariableOp_50бAssignVariableOp_51бAssignVariableOp_52бAssignVariableOp_53бAssignVariableOp_54бAssignVariableOp_55бAssignVariableOp_56бAssignVariableOp_57бAssignVariableOp_58бAssignVariableOp_59бAssignVariableOp_6бAssignVariableOp_60бAssignVariableOp_61бAssignVariableOp_62бAssignVariableOp_63бAssignVariableOp_64бAssignVariableOp_65бAssignVariableOp_66бAssignVariableOp_67бAssignVariableOp_68бAssignVariableOp_69бAssignVariableOp_7бAssignVariableOp_70бAssignVariableOp_71бAssignVariableOp_72бAssignVariableOp_73бAssignVariableOp_74бAssignVariableOp_75бAssignVariableOp_76бAssignVariableOp_77бAssignVariableOp_78бAssignVariableOp_79бAssignVariableOp_8бAssignVariableOp_80бAssignVariableOp_81бAssignVariableOp_82бAssignVariableOp_9ћ/
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:T*
dtype0*а.
valueќ.BЊ.TB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names╣
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:T*
dtype0*й
value│B░TB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesм
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Т
_output_shapesМ
л::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*b
dtypesX
V2T	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityг
AssignVariableOpAssignVariableOp-assignvariableop_res_block_output_base_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1▓
AssignVariableOp_1AssignVariableOp-assignvariableop_1_res_block_output_base_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ц
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv2d_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Б
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv2d_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Д
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ц
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Д
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ц
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Д
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ц
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ф
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv2d_4_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Е
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv2d_4_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ф
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_5_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Е
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_5_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ф
AssignVariableOp_14AssignVariableOp#assignvariableop_14_conv2d_6_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Е
AssignVariableOp_15AssignVariableOp!assignvariableop_15_conv2d_6_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Г
AssignVariableOp_16AssignVariableOp%assignvariableop_16_value_conv_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ф
AssignVariableOp_17AssignVariableOp#assignvariableop_17_value_conv_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ф
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv2d_7_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Е
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv2d_7_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20е
AssignVariableOp_20AssignVariableOp assignvariableop_20_dense_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21д
AssignVariableOp_21AssignVariableOpassignvariableop_21_dense_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22»
AssignVariableOp_22AssignVariableOp'assignvariableop_22_value_output_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Г
AssignVariableOp_23AssignVariableOp%assignvariableop_23_value_output_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_24Ц
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_iterIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Д
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_beta_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Д
AssignVariableOp_26AssignVariableOpassignvariableop_26_adam_beta_2Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27д
AssignVariableOp_27AssignVariableOpassignvariableop_27_adam_decayIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28«
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_learning_rateIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29А
AssignVariableOp_29AssignVariableOpassignvariableop_29_totalIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30А
AssignVariableOp_30AssignVariableOpassignvariableop_30_countIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Б
AssignVariableOp_31AssignVariableOpassignvariableop_31_total_1Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Б
AssignVariableOp_32AssignVariableOpassignvariableop_32_count_1Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Б
AssignVariableOp_33AssignVariableOpassignvariableop_33_total_2Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Б
AssignVariableOp_34AssignVariableOpassignvariableop_34_count_2Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35┐
AssignVariableOp_35AssignVariableOp7assignvariableop_35_adam_res_block_output_base_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36й
AssignVariableOp_36AssignVariableOp5assignvariableop_36_adam_res_block_output_base_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37░
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_conv2d_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38«
AssignVariableOp_38AssignVariableOp&assignvariableop_38_adam_conv2d_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39▓
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_conv2d_1_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40░
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_conv2d_1_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41▓
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv2d_2_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42░
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv2d_2_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43▓
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_conv2d_3_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44░
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_conv2d_3_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45▓
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_conv2d_4_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46░
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_conv2d_4_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47▓
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_conv2d_5_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48░
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_conv2d_5_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49▓
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_conv2d_6_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50░
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_conv2d_6_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51┤
AssignVariableOp_51AssignVariableOp,assignvariableop_51_adam_value_conv_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52▓
AssignVariableOp_52AssignVariableOp*assignvariableop_52_adam_value_conv_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53▓
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_conv2d_7_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54░
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_conv2d_7_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55»
AssignVariableOp_55AssignVariableOp'assignvariableop_55_adam_dense_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56Г
AssignVariableOp_56AssignVariableOp%assignvariableop_56_adam_dense_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57Х
AssignVariableOp_57AssignVariableOp.assignvariableop_57_adam_value_output_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58┤
AssignVariableOp_58AssignVariableOp,assignvariableop_58_adam_value_output_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59┐
AssignVariableOp_59AssignVariableOp7assignvariableop_59_adam_res_block_output_base_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60й
AssignVariableOp_60AssignVariableOp5assignvariableop_60_adam_res_block_output_base_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61░
AssignVariableOp_61AssignVariableOp(assignvariableop_61_adam_conv2d_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62«
AssignVariableOp_62AssignVariableOp&assignvariableop_62_adam_conv2d_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63▓
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_conv2d_1_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64░
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_conv2d_1_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65▓
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_conv2d_2_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66░
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_conv2d_2_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67▓
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_conv2d_3_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68░
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_conv2d_3_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69▓
AssignVariableOp_69AssignVariableOp*assignvariableop_69_adam_conv2d_4_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70░
AssignVariableOp_70AssignVariableOp(assignvariableop_70_adam_conv2d_4_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71▓
AssignVariableOp_71AssignVariableOp*assignvariableop_71_adam_conv2d_5_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72░
AssignVariableOp_72AssignVariableOp(assignvariableop_72_adam_conv2d_5_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73▓
AssignVariableOp_73AssignVariableOp*assignvariableop_73_adam_conv2d_6_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74░
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adam_conv2d_6_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75┤
AssignVariableOp_75AssignVariableOp,assignvariableop_75_adam_value_conv_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76▓
AssignVariableOp_76AssignVariableOp*assignvariableop_76_adam_value_conv_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77▓
AssignVariableOp_77AssignVariableOp*assignvariableop_77_adam_conv2d_7_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78░
AssignVariableOp_78AssignVariableOp(assignvariableop_78_adam_conv2d_7_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79»
AssignVariableOp_79AssignVariableOp'assignvariableop_79_adam_dense_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80Г
AssignVariableOp_80AssignVariableOp%assignvariableop_80_adam_dense_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81Х
AssignVariableOp_81AssignVariableOp.assignvariableop_81_adam_value_output_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82┤
AssignVariableOp_82AssignVariableOp,assignvariableop_82_adam_value_output_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_829
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpђ
Identity_83Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_83f
Identity_84IdentityIdentity_83:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_84У
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_84Identity_84:output:0*й
_input_shapesФ
е: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
У
b
F__inference_flatten_1_layer_call_and_return_conditional_losses_1591817

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         └2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
А
Ъ
*__inference_conv2d_6_layer_call_fn_1593089

inputs!
unknown:@@
	unknown_0:@
identityѕбStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_15918052
StatefulPartitionedCallЃ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
э
P
4__inference_res_block_output_2_layer_call_fn_1593075

inputs
identity┌
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *X
fSRQ
O__inference_res_block_output_2_layer_call_and_return_conditional_losses_15917752
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
в
■
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1591690

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
в
■
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1591756

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
ё
З
B__inference_dense_layer_call_and_return_conditional_losses_1593181

inputs1
matmul_readvariableop_resource:	└ -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	└ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:          2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         └: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
Т
`
D__inference_flatten_layer_call_and_return_conditional_losses_1591858

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    ▄  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ▄2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ▄2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         ?:W S
/
_output_shapes
:         ?
 
_user_specified_nameinputs
Э
І
R__inference_res_block_output_base_layer_call_and_return_conditional_losses_1591624

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
в
■
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1591658

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
А
Ъ
*__inference_conv2d_4_layer_call_fn_1593027

inputs!
unknown:@@
	unknown_0:@
identityѕбStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_15917392
StatefulPartitionedCallЃ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Є
k
O__inference_res_block_output_0_layer_call_and_return_conditional_losses_1591677

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:         @2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
У
b
F__inference_flatten_1_layer_call_and_return_conditional_losses_1593150

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         └2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
в
■
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1592976

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Ь
f
J__inference_policy_output_layer_call_and_return_conditional_losses_1593191

inputs
identityX
SoftmaxSoftmaxinputs*
T0*(
_output_shapes
:         ▄2	
Softmaxf
IdentityIdentitySoftmax:softmax:0*
T0*(
_output_shapes
:         ▄2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ▄:P L
(
_output_shapes
:         ▄
 
_user_specified_nameinputs
А
Ъ
*__inference_conv2d_1_layer_call_fn_1592923

inputs!
unknown:@@
	unknown_0:@
identityѕбStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_15916582
StatefulPartitionedCallЃ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Є
k
O__inference_res_block_output_0_layer_call_and_return_conditional_losses_1592956

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:         @2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Є
k
O__inference_res_block_output_1_layer_call_and_return_conditional_losses_1591726

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:         @2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
§

Щ
I__inference_value_output_layer_call_and_return_conditional_losses_1591871

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:         2
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
в
■
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1591805

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
еѕ
ѓ
B__inference_model_layer_call_and_return_conditional_losses_1592775

inputsN
4res_block_output_base_conv2d_readvariableop_resource:@C
5res_block_output_base_biasadd_readvariableop_resource:@?
%conv2d_conv2d_readvariableop_resource:@@4
&conv2d_biasadd_readvariableop_resource:@A
'conv2d_1_conv2d_readvariableop_resource:@@6
(conv2d_1_biasadd_readvariableop_resource:@A
'conv2d_2_conv2d_readvariableop_resource:@@6
(conv2d_2_biasadd_readvariableop_resource:@A
'conv2d_3_conv2d_readvariableop_resource:@@6
(conv2d_3_biasadd_readvariableop_resource:@A
'conv2d_4_conv2d_readvariableop_resource:@@6
(conv2d_4_biasadd_readvariableop_resource:@A
'conv2d_5_conv2d_readvariableop_resource:@@6
(conv2d_5_biasadd_readvariableop_resource:@C
)value_conv_conv2d_readvariableop_resource:@8
*value_conv_biasadd_readvariableop_resource:A
'conv2d_6_conv2d_readvariableop_resource:@@6
(conv2d_6_biasadd_readvariableop_resource:@A
'conv2d_7_conv2d_readvariableop_resource:@?6
(conv2d_7_biasadd_readvariableop_resource:?7
$dense_matmul_readvariableop_resource:	└ 3
%dense_biasadd_readvariableop_resource: =
+value_output_matmul_readvariableop_resource: :
,value_output_biasadd_readvariableop_resource:
identity

identity_1ѕбconv2d/BiasAdd/ReadVariableOpбconv2d/Conv2D/ReadVariableOpбconv2d_1/BiasAdd/ReadVariableOpбconv2d_1/Conv2D/ReadVariableOpбconv2d_2/BiasAdd/ReadVariableOpбconv2d_2/Conv2D/ReadVariableOpбconv2d_3/BiasAdd/ReadVariableOpбconv2d_3/Conv2D/ReadVariableOpбconv2d_4/BiasAdd/ReadVariableOpбconv2d_4/Conv2D/ReadVariableOpбconv2d_5/BiasAdd/ReadVariableOpбconv2d_5/Conv2D/ReadVariableOpбconv2d_6/BiasAdd/ReadVariableOpбconv2d_6/Conv2D/ReadVariableOpбconv2d_7/BiasAdd/ReadVariableOpбconv2d_7/Conv2D/ReadVariableOpбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpб,res_block_output_base/BiasAdd/ReadVariableOpб+res_block_output_base/Conv2D/ReadVariableOpб!value_conv/BiasAdd/ReadVariableOpб value_conv/Conv2D/ReadVariableOpб#value_output/BiasAdd/ReadVariableOpб"value_output/MatMul/ReadVariableOpО
+res_block_output_base/Conv2D/ReadVariableOpReadVariableOp4res_block_output_base_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02-
+res_block_output_base/Conv2D/ReadVariableOpт
res_block_output_base/Conv2DConv2Dinputs3res_block_output_base/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
res_block_output_base/Conv2D╬
,res_block_output_base/BiasAdd/ReadVariableOpReadVariableOp5res_block_output_base_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,res_block_output_base/BiasAdd/ReadVariableOpЯ
res_block_output_base/BiasAddBiasAdd%res_block_output_base/Conv2D:output:04res_block_output_base/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
res_block_output_base/BiasAddб
res_block_output_base/ReluRelu&res_block_output_base/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
res_block_output_base/Reluф
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
conv2d/Conv2D/ReadVariableOp┌
conv2d/Conv2DConv2D(res_block_output_base/Relu:activations:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d/Conv2DА
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOpц
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d/Relu░
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOpЛ
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_1/Conv2DД
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpг
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d_1/Reluю
add/addAddV2(res_block_output_base/Relu:activations:0conv2d_1/Relu:activations:0*
T0*/
_output_shapes
:         @2	
add/addЂ
res_block_output_0/ReluReluadd/add:z:0*
T0*/
_output_shapes
:         @2
res_block_output_0/Relu░
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOpП
conv2d_2/Conv2DConv2D%res_block_output_0/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_2/Conv2DД
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpг
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d_2/Relu░
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOpМ
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_3/Conv2DД
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOpг
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d_3/ReluЮ
	add_1/addAddV2%res_block_output_0/Relu:activations:0conv2d_3/Relu:activations:0*
T0*/
_output_shapes
:         @2
	add_1/addЃ
res_block_output_1/ReluReluadd_1/add:z:0*
T0*/
_output_shapes
:         @2
res_block_output_1/Relu░
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_4/Conv2D/ReadVariableOpП
conv2d_4/Conv2DConv2D%res_block_output_1/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_4/Conv2DД
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_4/BiasAdd/ReadVariableOpг
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_4/BiasAdd{
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d_4/Relu░
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_5/Conv2D/ReadVariableOpМ
conv2d_5/Conv2DConv2Dconv2d_4/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_5/Conv2DД
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_5/BiasAdd/ReadVariableOpг
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_5/BiasAdd{
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d_5/ReluЮ
	add_2/addAddV2%res_block_output_1/Relu:activations:0conv2d_5/Relu:activations:0*
T0*/
_output_shapes
:         @2
	add_2/addЃ
res_block_output_2/ReluReluadd_2/add:z:0*
T0*/
_output_shapes
:         @2
res_block_output_2/ReluХ
 value_conv/Conv2D/ReadVariableOpReadVariableOp)value_conv_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02"
 value_conv/Conv2D/ReadVariableOpс
value_conv/Conv2DConv2D%res_block_output_2/Relu:activations:0(value_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
2
value_conv/Conv2DГ
!value_conv/BiasAdd/ReadVariableOpReadVariableOp*value_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!value_conv/BiasAdd/ReadVariableOp┤
value_conv/BiasAddBiasAddvalue_conv/Conv2D:output:0)value_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
value_conv/BiasAddЂ
value_conv/ReluReluvalue_conv/BiasAdd:output:0*
T0*/
_output_shapes
:         2
value_conv/Relu░
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_6/Conv2D/ReadVariableOpП
conv2d_6/Conv2DConv2D%res_block_output_2/Relu:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_6/Conv2DД
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_6/BiasAdd/ReadVariableOpг
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_6/BiasAdd{
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d_6/Relus
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  2
flatten_1/ConstЮ
flatten_1/ReshapeReshapevalue_conv/Relu:activations:0flatten_1/Const:output:0*
T0*(
_output_shapes
:         └2
flatten_1/Reshape░
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@?*
dtype02 
conv2d_7/Conv2D/ReadVariableOpМ
conv2d_7/Conv2DConv2Dconv2d_6/Relu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ?*
paddingSAME*
strides
2
conv2d_7/Conv2DД
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:?*
dtype02!
conv2d_7/BiasAdd/ReadVariableOpг
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ?2
conv2d_7/BiasAddа
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	└ *
dtype02
dense/MatMul/ReadVariableOpЎ
dense/MatMulMatMulflatten_1/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense/MatMulъ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
dense/BiasAdd/ReadVariableOpЎ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:          2

dense/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    ▄  2
flatten/ConstЊ
flatten/ReshapeReshapeconv2d_7/BiasAdd:output:0flatten/Const:output:0*
T0*(
_output_shapes
:         ▄2
flatten/Reshape┤
"value_output/MatMul/ReadVariableOpReadVariableOp+value_output_matmul_readvariableop_resource*
_output_shapes

: *
dtype02$
"value_output/MatMul/ReadVariableOpг
value_output/MatMulMatMuldense/Relu:activations:0*value_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
value_output/MatMul│
#value_output/BiasAdd/ReadVariableOpReadVariableOp,value_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#value_output/BiasAdd/ReadVariableOpх
value_output/BiasAddBiasAddvalue_output/MatMul:product:0+value_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
value_output/BiasAdd
value_output/TanhTanhvalue_output/BiasAdd:output:0*
T0*'
_output_shapes
:         2
value_output/Tanhє
policy_output/SoftmaxSoftmaxflatten/Reshape:output:0*
T0*(
_output_shapes
:         ▄2
policy_output/Softmax{
IdentityIdentitypolicy_output/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:         ▄2

Identityt

Identity_1Identityvalue_output/Tanh:y:0^NoOp*
T0*'
_output_shapes
:         2

Identity_1ј
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp-^res_block_output_base/BiasAdd/ReadVariableOp,^res_block_output_base/Conv2D/ReadVariableOp"^value_conv/BiasAdd/ReadVariableOp!^value_conv/Conv2D/ReadVariableOp$^value_output/BiasAdd/ReadVariableOp#^value_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:         : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2\
,res_block_output_base/BiasAdd/ReadVariableOp,res_block_output_base/BiasAdd/ReadVariableOp2Z
+res_block_output_base/Conv2D/ReadVariableOp+res_block_output_base/Conv2D/ReadVariableOp2F
!value_conv/BiasAdd/ReadVariableOp!value_conv/BiasAdd/ReadVariableOp2D
 value_conv/Conv2D/ReadVariableOp value_conv/Conv2D/ReadVariableOp2J
#value_output/BiasAdd/ReadVariableOp#value_output/BiasAdd/ReadVariableOp2H
"value_output/MatMul/ReadVariableOp"value_output/MatMul/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Щ
l
B__inference_add_1_layer_call_and_return_conditional_losses_1591719

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:         @2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         @:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs:WS
/
_output_shapes
:         @
 
_user_specified_nameinputs
о
Н
'__inference_model_layer_call_fn_1591939
input_1!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@#
	unknown_7:@@
	unknown_8:@#
	unknown_9:@@

unknown_10:@$

unknown_11:@@

unknown_12:@$

unknown_13:@

unknown_14:$

unknown_15:@@

unknown_16:@$

unknown_17:@?

unknown_18:?

unknown_19:	└ 

unknown_20: 

unknown_21: 

unknown_22:
identity

identity_1ѕбStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *;
_output_shapes)
':         ▄:         *:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_15918862
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ▄2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:         : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_1
╗
г
7__inference_res_block_output_base_layer_call_fn_1592883

inputs!
unknown:@
	unknown_0:@
identityѕбStatefulPartitionedCallЈ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *[
fVRT
R__inference_res_block_output_base_layer_call_and_return_conditional_losses_15916242
StatefulPartitionedCallЃ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Э
j
@__inference_add_layer_call_and_return_conditional_losses_1591670

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:         @2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         @:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs:WS
/
_output_shapes
:         @
 
_user_specified_nameinputs
в
■
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1591739

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
ь
ђ
G__inference_value_conv_layer_call_and_return_conditional_losses_1591788

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Є
■
E__inference_conv2d_7_layer_call_and_return_conditional_losses_1593139

inputs8
conv2d_readvariableop_resource:@?-
biasadd_readvariableop_resource:?
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ?*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:?*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ?2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         ?2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
в
■
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1592996

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
в
■
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1593038

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Кб
Ћ"
 __inference__traced_save_1593484
file_prefix;
7savev2_res_block_output_base_kernel_read_readvariableop9
5savev2_res_block_output_base_bias_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop0
,savev2_value_conv_kernel_read_readvariableop.
*savev2_value_conv_bias_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop2
.savev2_value_output_kernel_read_readvariableop0
,savev2_value_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableopB
>savev2_adam_res_block_output_base_kernel_m_read_readvariableop@
<savev2_adam_res_block_output_base_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableop5
1savev2_adam_conv2d_4_kernel_m_read_readvariableop3
/savev2_adam_conv2d_4_bias_m_read_readvariableop5
1savev2_adam_conv2d_5_kernel_m_read_readvariableop3
/savev2_adam_conv2d_5_bias_m_read_readvariableop5
1savev2_adam_conv2d_6_kernel_m_read_readvariableop3
/savev2_adam_conv2d_6_bias_m_read_readvariableop7
3savev2_adam_value_conv_kernel_m_read_readvariableop5
1savev2_adam_value_conv_bias_m_read_readvariableop5
1savev2_adam_conv2d_7_kernel_m_read_readvariableop3
/savev2_adam_conv2d_7_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop9
5savev2_adam_value_output_kernel_m_read_readvariableop7
3savev2_adam_value_output_bias_m_read_readvariableopB
>savev2_adam_res_block_output_base_kernel_v_read_readvariableop@
<savev2_adam_res_block_output_base_bias_v_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableop5
1savev2_adam_conv2d_4_kernel_v_read_readvariableop3
/savev2_adam_conv2d_4_bias_v_read_readvariableop5
1savev2_adam_conv2d_5_kernel_v_read_readvariableop3
/savev2_adam_conv2d_5_bias_v_read_readvariableop5
1savev2_adam_conv2d_6_kernel_v_read_readvariableop3
/savev2_adam_conv2d_6_bias_v_read_readvariableop7
3savev2_adam_value_conv_kernel_v_read_readvariableop5
1savev2_adam_value_conv_bias_v_read_readvariableop5
1savev2_adam_conv2d_7_kernel_v_read_readvariableop3
/savev2_adam_conv2d_7_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop9
5savev2_adam_value_output_kernel_v_read_readvariableop7
3savev2_adam_value_output_bias_v_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1І
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameј/
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:T*
dtype0*а.
valueќ.BЊ.TB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names│
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:T*
dtype0*й
value│B░TB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesж 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:07savev2_res_block_output_base_kernel_read_readvariableop5savev2_res_block_output_base_bias_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop,savev2_value_conv_kernel_read_readvariableop*savev2_value_conv_bias_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop.savev2_value_output_kernel_read_readvariableop,savev2_value_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop>savev2_adam_res_block_output_base_kernel_m_read_readvariableop<savev2_adam_res_block_output_base_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop1savev2_adam_conv2d_4_kernel_m_read_readvariableop/savev2_adam_conv2d_4_bias_m_read_readvariableop1savev2_adam_conv2d_5_kernel_m_read_readvariableop/savev2_adam_conv2d_5_bias_m_read_readvariableop1savev2_adam_conv2d_6_kernel_m_read_readvariableop/savev2_adam_conv2d_6_bias_m_read_readvariableop3savev2_adam_value_conv_kernel_m_read_readvariableop1savev2_adam_value_conv_bias_m_read_readvariableop1savev2_adam_conv2d_7_kernel_m_read_readvariableop/savev2_adam_conv2d_7_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop5savev2_adam_value_output_kernel_m_read_readvariableop3savev2_adam_value_output_bias_m_read_readvariableop>savev2_adam_res_block_output_base_kernel_v_read_readvariableop<savev2_adam_res_block_output_base_bias_v_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop1savev2_adam_conv2d_4_kernel_v_read_readvariableop/savev2_adam_conv2d_4_bias_v_read_readvariableop1savev2_adam_conv2d_5_kernel_v_read_readvariableop/savev2_adam_conv2d_5_bias_v_read_readvariableop1savev2_adam_conv2d_6_kernel_v_read_readvariableop/savev2_adam_conv2d_6_bias_v_read_readvariableop3savev2_adam_value_conv_kernel_v_read_readvariableop1savev2_adam_value_conv_bias_v_read_readvariableop1savev2_adam_conv2d_7_kernel_v_read_readvariableop/savev2_adam_conv2d_7_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop5savev2_adam_value_output_kernel_v_read_readvariableop3savev2_adam_value_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *b
dtypesX
V2T	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*Р
_input_shapesл
═: :@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@::@?:?:	└ : : :: : : : : : : : : : : :@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@::@?:?:	└ : : ::@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@::@?:?:	└ : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,	(
&
_output_shapes
:@@: 


_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@: 

_output_shapes
::,(
&
_output_shapes
:@?: 

_output_shapes
:?:%!

_output_shapes
:	└ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :,$(
&
_output_shapes
:@: %

_output_shapes
:@:,&(
&
_output_shapes
:@@: '

_output_shapes
:@:,((
&
_output_shapes
:@@: )

_output_shapes
:@:,*(
&
_output_shapes
:@@: +

_output_shapes
:@:,,(
&
_output_shapes
:@@: -

_output_shapes
:@:,.(
&
_output_shapes
:@@: /

_output_shapes
:@:,0(
&
_output_shapes
:@@: 1

_output_shapes
:@:,2(
&
_output_shapes
:@@: 3

_output_shapes
:@:,4(
&
_output_shapes
:@: 5

_output_shapes
::,6(
&
_output_shapes
:@?: 7

_output_shapes
:?:%8!

_output_shapes
:	└ : 9

_output_shapes
: :$: 

_output_shapes

: : ;

_output_shapes
::,<(
&
_output_shapes
:@: =

_output_shapes
:@:,>(
&
_output_shapes
:@@: ?

_output_shapes
:@:,@(
&
_output_shapes
:@@: A

_output_shapes
:@:,B(
&
_output_shapes
:@@: C

_output_shapes
:@:,D(
&
_output_shapes
:@@: E

_output_shapes
:@:,F(
&
_output_shapes
:@@: G

_output_shapes
:@:,H(
&
_output_shapes
:@@: I

_output_shapes
:@:,J(
&
_output_shapes
:@@: K

_output_shapes
:@:,L(
&
_output_shapes
:@: M

_output_shapes
::,N(
&
_output_shapes
:@?: O

_output_shapes
:?:%P!

_output_shapes
:	└ : Q

_output_shapes
: :$R 

_output_shapes

: : S

_output_shapes
::T

_output_shapes
: 
М
н
'__inference_model_layer_call_fn_1592621

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@#
	unknown_7:@@
	unknown_8:@#
	unknown_9:@@

unknown_10:@$

unknown_11:@@

unknown_12:@$

unknown_13:@

unknown_14:$

unknown_15:@@

unknown_16:@$

unknown_17:@?

unknown_18:?

unknown_19:	└ 

unknown_20: 

unknown_21: 

unknown_22:
identity

identity_1ѕбStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *;
_output_shapes)
':         ▄:         *:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_15918862
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ▄2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:         : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ж
Ч
C__inference_conv2d_layer_call_and_return_conditional_losses_1591641

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
о
Н
'__inference_model_layer_call_fn_1592355
input_1!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@#
	unknown_7:@@
	unknown_8:@#
	unknown_9:@@

unknown_10:@$

unknown_11:@@

unknown_12:@$

unknown_13:@

unknown_14:$

unknown_15:@@

unknown_16:@$

unknown_17:@?

unknown_18:?

unknown_19:	└ 

unknown_20: 

unknown_21: 

unknown_22:
identity

identity_1ѕбStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *;
_output_shapes)
':         ▄:         *:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_15922472
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ▄2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:         : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_1
О
G
+__inference_flatten_1_layer_call_fn_1593144

inputs
identity╩
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_15918172
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Є
k
O__inference_res_block_output_1_layer_call_and_return_conditional_losses_1593018

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:         @2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
ё
З
B__inference_dense_layer_call_and_return_conditional_losses_1591846

inputs1
matmul_readvariableop_resource:	└ -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	└ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:          2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         └: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
еѕ
ѓ
B__inference_model_layer_call_and_return_conditional_losses_1592874

inputsN
4res_block_output_base_conv2d_readvariableop_resource:@C
5res_block_output_base_biasadd_readvariableop_resource:@?
%conv2d_conv2d_readvariableop_resource:@@4
&conv2d_biasadd_readvariableop_resource:@A
'conv2d_1_conv2d_readvariableop_resource:@@6
(conv2d_1_biasadd_readvariableop_resource:@A
'conv2d_2_conv2d_readvariableop_resource:@@6
(conv2d_2_biasadd_readvariableop_resource:@A
'conv2d_3_conv2d_readvariableop_resource:@@6
(conv2d_3_biasadd_readvariableop_resource:@A
'conv2d_4_conv2d_readvariableop_resource:@@6
(conv2d_4_biasadd_readvariableop_resource:@A
'conv2d_5_conv2d_readvariableop_resource:@@6
(conv2d_5_biasadd_readvariableop_resource:@C
)value_conv_conv2d_readvariableop_resource:@8
*value_conv_biasadd_readvariableop_resource:A
'conv2d_6_conv2d_readvariableop_resource:@@6
(conv2d_6_biasadd_readvariableop_resource:@A
'conv2d_7_conv2d_readvariableop_resource:@?6
(conv2d_7_biasadd_readvariableop_resource:?7
$dense_matmul_readvariableop_resource:	└ 3
%dense_biasadd_readvariableop_resource: =
+value_output_matmul_readvariableop_resource: :
,value_output_biasadd_readvariableop_resource:
identity

identity_1ѕбconv2d/BiasAdd/ReadVariableOpбconv2d/Conv2D/ReadVariableOpбconv2d_1/BiasAdd/ReadVariableOpбconv2d_1/Conv2D/ReadVariableOpбconv2d_2/BiasAdd/ReadVariableOpбconv2d_2/Conv2D/ReadVariableOpбconv2d_3/BiasAdd/ReadVariableOpбconv2d_3/Conv2D/ReadVariableOpбconv2d_4/BiasAdd/ReadVariableOpбconv2d_4/Conv2D/ReadVariableOpбconv2d_5/BiasAdd/ReadVariableOpбconv2d_5/Conv2D/ReadVariableOpбconv2d_6/BiasAdd/ReadVariableOpбconv2d_6/Conv2D/ReadVariableOpбconv2d_7/BiasAdd/ReadVariableOpбconv2d_7/Conv2D/ReadVariableOpбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpб,res_block_output_base/BiasAdd/ReadVariableOpб+res_block_output_base/Conv2D/ReadVariableOpб!value_conv/BiasAdd/ReadVariableOpб value_conv/Conv2D/ReadVariableOpб#value_output/BiasAdd/ReadVariableOpб"value_output/MatMul/ReadVariableOpО
+res_block_output_base/Conv2D/ReadVariableOpReadVariableOp4res_block_output_base_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02-
+res_block_output_base/Conv2D/ReadVariableOpт
res_block_output_base/Conv2DConv2Dinputs3res_block_output_base/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
res_block_output_base/Conv2D╬
,res_block_output_base/BiasAdd/ReadVariableOpReadVariableOp5res_block_output_base_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,res_block_output_base/BiasAdd/ReadVariableOpЯ
res_block_output_base/BiasAddBiasAdd%res_block_output_base/Conv2D:output:04res_block_output_base/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
res_block_output_base/BiasAddб
res_block_output_base/ReluRelu&res_block_output_base/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
res_block_output_base/Reluф
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
conv2d/Conv2D/ReadVariableOp┌
conv2d/Conv2DConv2D(res_block_output_base/Relu:activations:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d/Conv2DА
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOpц
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d/Relu░
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOpЛ
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_1/Conv2DД
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpг
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d_1/Reluю
add/addAddV2(res_block_output_base/Relu:activations:0conv2d_1/Relu:activations:0*
T0*/
_output_shapes
:         @2	
add/addЂ
res_block_output_0/ReluReluadd/add:z:0*
T0*/
_output_shapes
:         @2
res_block_output_0/Relu░
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOpП
conv2d_2/Conv2DConv2D%res_block_output_0/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_2/Conv2DД
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpг
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d_2/Relu░
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOpМ
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_3/Conv2DД
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOpг
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d_3/ReluЮ
	add_1/addAddV2%res_block_output_0/Relu:activations:0conv2d_3/Relu:activations:0*
T0*/
_output_shapes
:         @2
	add_1/addЃ
res_block_output_1/ReluReluadd_1/add:z:0*
T0*/
_output_shapes
:         @2
res_block_output_1/Relu░
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_4/Conv2D/ReadVariableOpП
conv2d_4/Conv2DConv2D%res_block_output_1/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_4/Conv2DД
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_4/BiasAdd/ReadVariableOpг
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_4/BiasAdd{
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d_4/Relu░
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_5/Conv2D/ReadVariableOpМ
conv2d_5/Conv2DConv2Dconv2d_4/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_5/Conv2DД
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_5/BiasAdd/ReadVariableOpг
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_5/BiasAdd{
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d_5/ReluЮ
	add_2/addAddV2%res_block_output_1/Relu:activations:0conv2d_5/Relu:activations:0*
T0*/
_output_shapes
:         @2
	add_2/addЃ
res_block_output_2/ReluReluadd_2/add:z:0*
T0*/
_output_shapes
:         @2
res_block_output_2/ReluХ
 value_conv/Conv2D/ReadVariableOpReadVariableOp)value_conv_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02"
 value_conv/Conv2D/ReadVariableOpс
value_conv/Conv2DConv2D%res_block_output_2/Relu:activations:0(value_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
2
value_conv/Conv2DГ
!value_conv/BiasAdd/ReadVariableOpReadVariableOp*value_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!value_conv/BiasAdd/ReadVariableOp┤
value_conv/BiasAddBiasAddvalue_conv/Conv2D:output:0)value_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
value_conv/BiasAddЂ
value_conv/ReluReluvalue_conv/BiasAdd:output:0*
T0*/
_output_shapes
:         2
value_conv/Relu░
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_6/Conv2D/ReadVariableOpП
conv2d_6/Conv2DConv2D%res_block_output_2/Relu:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_6/Conv2DД
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_6/BiasAdd/ReadVariableOpг
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_6/BiasAdd{
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d_6/Relus
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  2
flatten_1/ConstЮ
flatten_1/ReshapeReshapevalue_conv/Relu:activations:0flatten_1/Const:output:0*
T0*(
_output_shapes
:         └2
flatten_1/Reshape░
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@?*
dtype02 
conv2d_7/Conv2D/ReadVariableOpМ
conv2d_7/Conv2DConv2Dconv2d_6/Relu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ?*
paddingSAME*
strides
2
conv2d_7/Conv2DД
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:?*
dtype02!
conv2d_7/BiasAdd/ReadVariableOpг
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ?2
conv2d_7/BiasAddа
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	└ *
dtype02
dense/MatMul/ReadVariableOpЎ
dense/MatMulMatMulflatten_1/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense/MatMulъ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
dense/BiasAdd/ReadVariableOpЎ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:          2

dense/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    ▄  2
flatten/ConstЊ
flatten/ReshapeReshapeconv2d_7/BiasAdd:output:0flatten/Const:output:0*
T0*(
_output_shapes
:         ▄2
flatten/Reshape┤
"value_output/MatMul/ReadVariableOpReadVariableOp+value_output_matmul_readvariableop_resource*
_output_shapes

: *
dtype02$
"value_output/MatMul/ReadVariableOpг
value_output/MatMulMatMuldense/Relu:activations:0*value_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
value_output/MatMul│
#value_output/BiasAdd/ReadVariableOpReadVariableOp,value_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#value_output/BiasAdd/ReadVariableOpх
value_output/BiasAddBiasAddvalue_output/MatMul:product:0+value_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
value_output/BiasAdd
value_output/TanhTanhvalue_output/BiasAdd:output:0*
T0*'
_output_shapes
:         2
value_output/Tanhє
policy_output/SoftmaxSoftmaxflatten/Reshape:output:0*
T0*(
_output_shapes
:         ▄2
policy_output/Softmax{
IdentityIdentitypolicy_output/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:         ▄2

Identityt

Identity_1Identityvalue_output/Tanh:y:0^NoOp*
T0*'
_output_shapes
:         2

Identity_1ј
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp-^res_block_output_base/BiasAdd/ReadVariableOp,^res_block_output_base/Conv2D/ReadVariableOp"^value_conv/BiasAdd/ReadVariableOp!^value_conv/Conv2D/ReadVariableOp$^value_output/BiasAdd/ReadVariableOp#^value_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:         : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2\
,res_block_output_base/BiasAdd/ReadVariableOp,res_block_output_base/BiasAdd/ReadVariableOp2Z
+res_block_output_base/Conv2D/ReadVariableOp+res_block_output_base/Conv2D/ReadVariableOp2F
!value_conv/BiasAdd/ReadVariableOp!value_conv/BiasAdd/ReadVariableOp2D
 value_conv/Conv2D/ReadVariableOp value_conv/Conv2D/ReadVariableOp2J
#value_output/BiasAdd/ReadVariableOp#value_output/BiasAdd/ReadVariableOp2H
"value_output/MatMul/ReadVariableOp"value_output/MatMul/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ж
Ч
C__inference_conv2d_layer_call_and_return_conditional_losses_1592914

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Є
■
E__inference_conv2d_7_layer_call_and_return_conditional_losses_1591829

inputs8
conv2d_readvariableop_resource:@?-
biasadd_readvariableop_resource:?
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ?*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:?*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ?2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         ?2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
А
Ъ
*__inference_conv2d_5_layer_call_fn_1593047

inputs!
unknown:@@
	unknown_0:@
identityѕбStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_15917562
StatefulPartitionedCallЃ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Ь
f
J__inference_policy_output_layer_call_and_return_conditional_losses_1591882

inputs
identityX
SoftmaxSoftmaxinputs*
T0*(
_output_shapes
:         ▄2	
Softmaxf
IdentityIdentitySoftmax:softmax:0*
T0*(
_output_shapes
:         ▄2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ▄:P L
(
_output_shapes
:         ▄
 
_user_specified_nameinputs
Эў
Ѓ
"__inference__wrapped_model_1591606
input_1T
:model_res_block_output_base_conv2d_readvariableop_resource:@I
;model_res_block_output_base_biasadd_readvariableop_resource:@E
+model_conv2d_conv2d_readvariableop_resource:@@:
,model_conv2d_biasadd_readvariableop_resource:@G
-model_conv2d_1_conv2d_readvariableop_resource:@@<
.model_conv2d_1_biasadd_readvariableop_resource:@G
-model_conv2d_2_conv2d_readvariableop_resource:@@<
.model_conv2d_2_biasadd_readvariableop_resource:@G
-model_conv2d_3_conv2d_readvariableop_resource:@@<
.model_conv2d_3_biasadd_readvariableop_resource:@G
-model_conv2d_4_conv2d_readvariableop_resource:@@<
.model_conv2d_4_biasadd_readvariableop_resource:@G
-model_conv2d_5_conv2d_readvariableop_resource:@@<
.model_conv2d_5_biasadd_readvariableop_resource:@I
/model_value_conv_conv2d_readvariableop_resource:@>
0model_value_conv_biasadd_readvariableop_resource:G
-model_conv2d_6_conv2d_readvariableop_resource:@@<
.model_conv2d_6_biasadd_readvariableop_resource:@G
-model_conv2d_7_conv2d_readvariableop_resource:@?<
.model_conv2d_7_biasadd_readvariableop_resource:?=
*model_dense_matmul_readvariableop_resource:	└ 9
+model_dense_biasadd_readvariableop_resource: C
1model_value_output_matmul_readvariableop_resource: @
2model_value_output_biasadd_readvariableop_resource:
identity

identity_1ѕб#model/conv2d/BiasAdd/ReadVariableOpб"model/conv2d/Conv2D/ReadVariableOpб%model/conv2d_1/BiasAdd/ReadVariableOpб$model/conv2d_1/Conv2D/ReadVariableOpб%model/conv2d_2/BiasAdd/ReadVariableOpб$model/conv2d_2/Conv2D/ReadVariableOpб%model/conv2d_3/BiasAdd/ReadVariableOpб$model/conv2d_3/Conv2D/ReadVariableOpб%model/conv2d_4/BiasAdd/ReadVariableOpб$model/conv2d_4/Conv2D/ReadVariableOpб%model/conv2d_5/BiasAdd/ReadVariableOpб$model/conv2d_5/Conv2D/ReadVariableOpб%model/conv2d_6/BiasAdd/ReadVariableOpб$model/conv2d_6/Conv2D/ReadVariableOpб%model/conv2d_7/BiasAdd/ReadVariableOpб$model/conv2d_7/Conv2D/ReadVariableOpб"model/dense/BiasAdd/ReadVariableOpб!model/dense/MatMul/ReadVariableOpб2model/res_block_output_base/BiasAdd/ReadVariableOpб1model/res_block_output_base/Conv2D/ReadVariableOpб'model/value_conv/BiasAdd/ReadVariableOpб&model/value_conv/Conv2D/ReadVariableOpб)model/value_output/BiasAdd/ReadVariableOpб(model/value_output/MatMul/ReadVariableOpж
1model/res_block_output_base/Conv2D/ReadVariableOpReadVariableOp:model_res_block_output_base_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype023
1model/res_block_output_base/Conv2D/ReadVariableOpЭ
"model/res_block_output_base/Conv2DConv2Dinput_19model/res_block_output_base/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2$
"model/res_block_output_base/Conv2DЯ
2model/res_block_output_base/BiasAdd/ReadVariableOpReadVariableOp;model_res_block_output_base_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype024
2model/res_block_output_base/BiasAdd/ReadVariableOpЭ
#model/res_block_output_base/BiasAddBiasAdd+model/res_block_output_base/Conv2D:output:0:model/res_block_output_base/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2%
#model/res_block_output_base/BiasAdd┤
 model/res_block_output_base/ReluRelu,model/res_block_output_base/BiasAdd:output:0*
T0*/
_output_shapes
:         @2"
 model/res_block_output_base/Relu╝
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"model/conv2d/Conv2D/ReadVariableOpЫ
model/conv2d/Conv2DConv2D.model/res_block_output_base/Relu:activations:0*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
model/conv2d/Conv2D│
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#model/conv2d/BiasAdd/ReadVariableOp╝
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
model/conv2d/BiasAddЄ
model/conv2d/ReluRelumodel/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
model/conv2d/Relu┬
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02&
$model/conv2d_1/Conv2D/ReadVariableOpж
model/conv2d_1/Conv2DConv2Dmodel/conv2d/Relu:activations:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
model/conv2d_1/Conv2D╣
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/conv2d_1/BiasAdd/ReadVariableOp─
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
model/conv2d_1/BiasAddЇ
model/conv2d_1/ReluRelumodel/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
model/conv2d_1/Relu┤
model/add/addAddV2.model/res_block_output_base/Relu:activations:0!model/conv2d_1/Relu:activations:0*
T0*/
_output_shapes
:         @2
model/add/addЊ
model/res_block_output_0/ReluRelumodel/add/add:z:0*
T0*/
_output_shapes
:         @2
model/res_block_output_0/Relu┬
$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02&
$model/conv2d_2/Conv2D/ReadVariableOpш
model/conv2d_2/Conv2DConv2D+model/res_block_output_0/Relu:activations:0,model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
model/conv2d_2/Conv2D╣
%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/conv2d_2/BiasAdd/ReadVariableOp─
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
model/conv2d_2/BiasAddЇ
model/conv2d_2/ReluRelumodel/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
model/conv2d_2/Relu┬
$model/conv2d_3/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02&
$model/conv2d_3/Conv2D/ReadVariableOpв
model/conv2d_3/Conv2DConv2D!model/conv2d_2/Relu:activations:0,model/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
model/conv2d_3/Conv2D╣
%model/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/conv2d_3/BiasAdd/ReadVariableOp─
model/conv2d_3/BiasAddBiasAddmodel/conv2d_3/Conv2D:output:0-model/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
model/conv2d_3/BiasAddЇ
model/conv2d_3/ReluRelumodel/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
model/conv2d_3/Reluх
model/add_1/addAddV2+model/res_block_output_0/Relu:activations:0!model/conv2d_3/Relu:activations:0*
T0*/
_output_shapes
:         @2
model/add_1/addЋ
model/res_block_output_1/ReluRelumodel/add_1/add:z:0*
T0*/
_output_shapes
:         @2
model/res_block_output_1/Relu┬
$model/conv2d_4/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02&
$model/conv2d_4/Conv2D/ReadVariableOpш
model/conv2d_4/Conv2DConv2D+model/res_block_output_1/Relu:activations:0,model/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
model/conv2d_4/Conv2D╣
%model/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/conv2d_4/BiasAdd/ReadVariableOp─
model/conv2d_4/BiasAddBiasAddmodel/conv2d_4/Conv2D:output:0-model/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
model/conv2d_4/BiasAddЇ
model/conv2d_4/ReluRelumodel/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
model/conv2d_4/Relu┬
$model/conv2d_5/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02&
$model/conv2d_5/Conv2D/ReadVariableOpв
model/conv2d_5/Conv2DConv2D!model/conv2d_4/Relu:activations:0,model/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
model/conv2d_5/Conv2D╣
%model/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/conv2d_5/BiasAdd/ReadVariableOp─
model/conv2d_5/BiasAddBiasAddmodel/conv2d_5/Conv2D:output:0-model/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
model/conv2d_5/BiasAddЇ
model/conv2d_5/ReluRelumodel/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
model/conv2d_5/Reluх
model/add_2/addAddV2+model/res_block_output_1/Relu:activations:0!model/conv2d_5/Relu:activations:0*
T0*/
_output_shapes
:         @2
model/add_2/addЋ
model/res_block_output_2/ReluRelumodel/add_2/add:z:0*
T0*/
_output_shapes
:         @2
model/res_block_output_2/Relu╚
&model/value_conv/Conv2D/ReadVariableOpReadVariableOp/model_value_conv_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02(
&model/value_conv/Conv2D/ReadVariableOpч
model/value_conv/Conv2DConv2D+model/res_block_output_2/Relu:activations:0.model/value_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
2
model/value_conv/Conv2D┐
'model/value_conv/BiasAdd/ReadVariableOpReadVariableOp0model_value_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model/value_conv/BiasAdd/ReadVariableOp╠
model/value_conv/BiasAddBiasAdd model/value_conv/Conv2D:output:0/model/value_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
model/value_conv/BiasAddЊ
model/value_conv/ReluRelu!model/value_conv/BiasAdd:output:0*
T0*/
_output_shapes
:         2
model/value_conv/Relu┬
$model/conv2d_6/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02&
$model/conv2d_6/Conv2D/ReadVariableOpш
model/conv2d_6/Conv2DConv2D+model/res_block_output_2/Relu:activations:0,model/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
model/conv2d_6/Conv2D╣
%model/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/conv2d_6/BiasAdd/ReadVariableOp─
model/conv2d_6/BiasAddBiasAddmodel/conv2d_6/Conv2D:output:0-model/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
model/conv2d_6/BiasAddЇ
model/conv2d_6/ReluRelumodel/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
model/conv2d_6/Relu
model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  2
model/flatten_1/Constх
model/flatten_1/ReshapeReshape#model/value_conv/Relu:activations:0model/flatten_1/Const:output:0*
T0*(
_output_shapes
:         └2
model/flatten_1/Reshape┬
$model/conv2d_7/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@?*
dtype02&
$model/conv2d_7/Conv2D/ReadVariableOpв
model/conv2d_7/Conv2DConv2D!model/conv2d_6/Relu:activations:0,model/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ?*
paddingSAME*
strides
2
model/conv2d_7/Conv2D╣
%model/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:?*
dtype02'
%model/conv2d_7/BiasAdd/ReadVariableOp─
model/conv2d_7/BiasAddBiasAddmodel/conv2d_7/Conv2D:output:0-model/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ?2
model/conv2d_7/BiasAdd▓
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	└ *
dtype02#
!model/dense/MatMul/ReadVariableOp▒
model/dense/MatMulMatMul model/flatten_1/Reshape:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model/dense/MatMul░
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"model/dense/BiasAdd/ReadVariableOp▒
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model/dense/BiasAdd|
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:          2
model/dense/Relu{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    ▄  2
model/flatten/ConstФ
model/flatten/ReshapeReshapemodel/conv2d_7/BiasAdd:output:0model/flatten/Const:output:0*
T0*(
_output_shapes
:         ▄2
model/flatten/Reshapeк
(model/value_output/MatMul/ReadVariableOpReadVariableOp1model_value_output_matmul_readvariableop_resource*
_output_shapes

: *
dtype02*
(model/value_output/MatMul/ReadVariableOp─
model/value_output/MatMulMatMulmodel/dense/Relu:activations:00model/value_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model/value_output/MatMul┼
)model/value_output/BiasAdd/ReadVariableOpReadVariableOp2model_value_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)model/value_output/BiasAdd/ReadVariableOp═
model/value_output/BiasAddBiasAdd#model/value_output/MatMul:product:01model/value_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model/value_output/BiasAddЉ
model/value_output/TanhTanh#model/value_output/BiasAdd:output:0*
T0*'
_output_shapes
:         2
model/value_output/Tanhў
model/policy_output/SoftmaxSoftmaxmodel/flatten/Reshape:output:0*
T0*(
_output_shapes
:         ▄2
model/policy_output/SoftmaxЂ
IdentityIdentity%model/policy_output/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:         ▄2

Identityz

Identity_1Identitymodel/value_output/Tanh:y:0^NoOp*
T0*'
_output_shapes
:         2

Identity_1ъ
NoOpNoOp$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp&^model/conv2d_3/BiasAdd/ReadVariableOp%^model/conv2d_3/Conv2D/ReadVariableOp&^model/conv2d_4/BiasAdd/ReadVariableOp%^model/conv2d_4/Conv2D/ReadVariableOp&^model/conv2d_5/BiasAdd/ReadVariableOp%^model/conv2d_5/Conv2D/ReadVariableOp&^model/conv2d_6/BiasAdd/ReadVariableOp%^model/conv2d_6/Conv2D/ReadVariableOp&^model/conv2d_7/BiasAdd/ReadVariableOp%^model/conv2d_7/Conv2D/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp3^model/res_block_output_base/BiasAdd/ReadVariableOp2^model/res_block_output_base/Conv2D/ReadVariableOp(^model/value_conv/BiasAdd/ReadVariableOp'^model/value_conv/Conv2D/ReadVariableOp*^model/value_output/BiasAdd/ReadVariableOp)^model/value_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:         : : : : : : : : : : : : : : : : : : : : : : : : 2J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2N
%model/conv2d_2/BiasAdd/ReadVariableOp%model/conv2d_2/BiasAdd/ReadVariableOp2L
$model/conv2d_2/Conv2D/ReadVariableOp$model/conv2d_2/Conv2D/ReadVariableOp2N
%model/conv2d_3/BiasAdd/ReadVariableOp%model/conv2d_3/BiasAdd/ReadVariableOp2L
$model/conv2d_3/Conv2D/ReadVariableOp$model/conv2d_3/Conv2D/ReadVariableOp2N
%model/conv2d_4/BiasAdd/ReadVariableOp%model/conv2d_4/BiasAdd/ReadVariableOp2L
$model/conv2d_4/Conv2D/ReadVariableOp$model/conv2d_4/Conv2D/ReadVariableOp2N
%model/conv2d_5/BiasAdd/ReadVariableOp%model/conv2d_5/BiasAdd/ReadVariableOp2L
$model/conv2d_5/Conv2D/ReadVariableOp$model/conv2d_5/Conv2D/ReadVariableOp2N
%model/conv2d_6/BiasAdd/ReadVariableOp%model/conv2d_6/BiasAdd/ReadVariableOp2L
$model/conv2d_6/Conv2D/ReadVariableOp$model/conv2d_6/Conv2D/ReadVariableOp2N
%model/conv2d_7/BiasAdd/ReadVariableOp%model/conv2d_7/BiasAdd/ReadVariableOp2L
$model/conv2d_7/Conv2D/ReadVariableOp$model/conv2d_7/Conv2D/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2h
2model/res_block_output_base/BiasAdd/ReadVariableOp2model/res_block_output_base/BiasAdd/ReadVariableOp2f
1model/res_block_output_base/Conv2D/ReadVariableOp1model/res_block_output_base/Conv2D/ReadVariableOp2R
'model/value_conv/BiasAdd/ReadVariableOp'model/value_conv/BiasAdd/ReadVariableOp2P
&model/value_conv/Conv2D/ReadVariableOp&model/value_conv/Conv2D/ReadVariableOp2V
)model/value_output/BiasAdd/ReadVariableOp)model/value_output/BiasAdd/ReadVariableOp2T
(model/value_output/MatMul/ReadVariableOp(model/value_output/MatMul/ReadVariableOp:X T
/
_output_shapes
:         
!
_user_specified_name	input_1
в
■
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1593058

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
в
■
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1591707

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
ѓ
n
B__inference_add_1_layer_call_and_return_conditional_losses_1593008
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:         @2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         @:         @:Y U
/
_output_shapes
:         @
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         @
"
_user_specified_name
inputs/1
Є
k
O__inference_res_block_output_2_layer_call_and_return_conditional_losses_1593080

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:         @2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
в
■
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1593100

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
§

Щ
I__inference_value_output_layer_call_and_return_conditional_losses_1593211

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:         2
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Є
k
O__inference_res_block_output_2_layer_call_and_return_conditional_losses_1591775

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:         @2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Ы
S
'__inference_add_2_layer_call_fn_1593064
inputs_0
inputs_1
identity┌
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_add_2_layer_call_and_return_conditional_losses_15917682
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         @:         @:Y U
/
_output_shapes
:         @
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         @
"
_user_specified_name
inputs/1
А
Ъ
*__inference_conv2d_3_layer_call_fn_1592985

inputs!
unknown:@@
	unknown_0:@
identityѕбStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_15917072
StatefulPartitionedCallЃ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
М
E
)__inference_flatten_layer_call_fn_1593155

inputs
identity╚
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ▄* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_15918582
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ▄2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         ?:W S
/
_output_shapes
:         ?
 
_user_specified_nameinputs
Т
`
D__inference_flatten_layer_call_and_return_conditional_losses_1593161

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    ▄  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ▄2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ▄2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         ?:W S
/
_output_shapes
:         ?
 
_user_specified_nameinputs
Ю
Ю
(__inference_conv2d_layer_call_fn_1592903

inputs!
unknown:@@
	unknown_0:@
identityѕбStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_15916412
StatefulPartitionedCallЃ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Пb
└
B__inference_model_layer_call_and_return_conditional_losses_1591886

inputs7
res_block_output_base_1591625:@+
res_block_output_base_1591627:@(
conv2d_1591642:@@
conv2d_1591644:@*
conv2d_1_1591659:@@
conv2d_1_1591661:@*
conv2d_2_1591691:@@
conv2d_2_1591693:@*
conv2d_3_1591708:@@
conv2d_3_1591710:@*
conv2d_4_1591740:@@
conv2d_4_1591742:@*
conv2d_5_1591757:@@
conv2d_5_1591759:@,
value_conv_1591789:@ 
value_conv_1591791:*
conv2d_6_1591806:@@
conv2d_6_1591808:@*
conv2d_7_1591830:@?
conv2d_7_1591832:? 
dense_1591847:	└ 
dense_1591849: &
value_output_1591872: "
value_output_1591874:
identity

identity_1ѕбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallб conv2d_2/StatefulPartitionedCallб conv2d_3/StatefulPartitionedCallб conv2d_4/StatefulPartitionedCallб conv2d_5/StatefulPartitionedCallб conv2d_6/StatefulPartitionedCallб conv2d_7/StatefulPartitionedCallбdense/StatefulPartitionedCallб-res_block_output_base/StatefulPartitionedCallб"value_conv/StatefulPartitionedCallб$value_output/StatefulPartitionedCallт
-res_block_output_base/StatefulPartitionedCallStatefulPartitionedCallinputsres_block_output_base_1591625res_block_output_base_1591627*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *[
fVRT
R__inference_res_block_output_base_layer_call_and_return_conditional_losses_15916242/
-res_block_output_base/StatefulPartitionedCall╩
conv2d/StatefulPartitionedCallStatefulPartitionedCall6res_block_output_base/StatefulPartitionedCall:output:0conv2d_1591642conv2d_1591644*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_15916412 
conv2d/StatefulPartitionedCall┼
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_1591659conv2d_1_1591661*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_15916582"
 conv2d_1/StatefulPartitionedCall»
add/PartitionedCallPartitionedCall6res_block_output_base/StatefulPartitionedCall:output:0)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_15916702
add/PartitionedCallќ
"res_block_output_0/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *X
fSRQ
O__inference_res_block_output_0_layer_call_and_return_conditional_losses_15916772$
"res_block_output_0/PartitionedCall╔
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall+res_block_output_0/PartitionedCall:output:0conv2d_2_1591691conv2d_2_1591693*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_15916902"
 conv2d_2/StatefulPartitionedCallК
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_1591708conv2d_3_1591710*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_15917072"
 conv2d_3/StatefulPartitionedCallф
add_1/PartitionedCallPartitionedCall+res_block_output_0/PartitionedCall:output:0)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_15917192
add_1/PartitionedCallў
"res_block_output_1/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *X
fSRQ
O__inference_res_block_output_1_layer_call_and_return_conditional_losses_15917262$
"res_block_output_1/PartitionedCall╔
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall+res_block_output_1/PartitionedCall:output:0conv2d_4_1591740conv2d_4_1591742*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_15917392"
 conv2d_4/StatefulPartitionedCallК
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_1591757conv2d_5_1591759*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_15917562"
 conv2d_5/StatefulPartitionedCallф
add_2/PartitionedCallPartitionedCall+res_block_output_1/PartitionedCall:output:0)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_add_2_layer_call_and_return_conditional_losses_15917682
add_2/PartitionedCallў
"res_block_output_2/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *X
fSRQ
O__inference_res_block_output_2_layer_call_and_return_conditional_losses_15917752$
"res_block_output_2/PartitionedCallМ
"value_conv/StatefulPartitionedCallStatefulPartitionedCall+res_block_output_2/PartitionedCall:output:0value_conv_1591789value_conv_1591791*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *P
fKRI
G__inference_value_conv_layer_call_and_return_conditional_losses_15917882$
"value_conv/StatefulPartitionedCall╔
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall+res_block_output_2/PartitionedCall:output:0conv2d_6_1591806conv2d_6_1591808*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_15918052"
 conv2d_6/StatefulPartitionedCallЃ
flatten_1/PartitionedCallPartitionedCall+value_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_15918172
flatten_1/PartitionedCallК
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_1591830conv2d_7_1591832*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_15918292"
 conv2d_7/StatefulPartitionedCallЕ
dense/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1591847dense_1591849*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_15918462
dense/StatefulPartitionedCallч
flatten/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ▄* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_15918582
flatten/PartitionedCallл
$value_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0value_output_1591872value_output_1591874*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *R
fMRK
I__inference_value_output_layer_call_and_return_conditional_losses_15918712&
$value_output/StatefulPartitionedCallё
policy_output/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ▄* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__inference_policy_output_layer_call_and_return_conditional_losses_15918822
policy_output/PartitionedCallѓ
IdentityIdentity&policy_output/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ▄2

Identityї

Identity_1Identity-value_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity_1ђ
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall^dense/StatefulPartitionedCall.^res_block_output_base/StatefulPartitionedCall#^value_conv/StatefulPartitionedCall%^value_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:         : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2^
-res_block_output_base/StatefulPartitionedCall-res_block_output_base/StatefulPartitionedCall2H
"value_conv/StatefulPartitionedCall"value_conv/StatefulPartitionedCall2L
$value_output/StatefulPartitionedCall$value_output/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
А
Ъ
*__inference_conv2d_2_layer_call_fn_1592965

inputs!
unknown:@@
	unknown_0:@
identityѕбStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_15916902
StatefulPartitionedCallЃ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
А
Ъ
*__inference_conv2d_7_layer_call_fn_1593129

inputs!
unknown:@?
	unknown_0:?
identityѕбStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_15918292
StatefulPartitionedCallЃ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         ?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
в
■
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1592934

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Яb
┴
B__inference_model_layer_call_and_return_conditional_losses_1592429
input_17
res_block_output_base_1592358:@+
res_block_output_base_1592360:@(
conv2d_1592363:@@
conv2d_1592365:@*
conv2d_1_1592368:@@
conv2d_1_1592370:@*
conv2d_2_1592375:@@
conv2d_2_1592377:@*
conv2d_3_1592380:@@
conv2d_3_1592382:@*
conv2d_4_1592387:@@
conv2d_4_1592389:@*
conv2d_5_1592392:@@
conv2d_5_1592394:@,
value_conv_1592399:@ 
value_conv_1592401:*
conv2d_6_1592404:@@
conv2d_6_1592406:@*
conv2d_7_1592410:@?
conv2d_7_1592412:? 
dense_1592415:	└ 
dense_1592417: &
value_output_1592421: "
value_output_1592423:
identity

identity_1ѕбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallб conv2d_2/StatefulPartitionedCallб conv2d_3/StatefulPartitionedCallб conv2d_4/StatefulPartitionedCallб conv2d_5/StatefulPartitionedCallб conv2d_6/StatefulPartitionedCallб conv2d_7/StatefulPartitionedCallбdense/StatefulPartitionedCallб-res_block_output_base/StatefulPartitionedCallб"value_conv/StatefulPartitionedCallб$value_output/StatefulPartitionedCallТ
-res_block_output_base/StatefulPartitionedCallStatefulPartitionedCallinput_1res_block_output_base_1592358res_block_output_base_1592360*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *[
fVRT
R__inference_res_block_output_base_layer_call_and_return_conditional_losses_15916242/
-res_block_output_base/StatefulPartitionedCall╩
conv2d/StatefulPartitionedCallStatefulPartitionedCall6res_block_output_base/StatefulPartitionedCall:output:0conv2d_1592363conv2d_1592365*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_15916412 
conv2d/StatefulPartitionedCall┼
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_1592368conv2d_1_1592370*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_15916582"
 conv2d_1/StatefulPartitionedCall»
add/PartitionedCallPartitionedCall6res_block_output_base/StatefulPartitionedCall:output:0)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_15916702
add/PartitionedCallќ
"res_block_output_0/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *X
fSRQ
O__inference_res_block_output_0_layer_call_and_return_conditional_losses_15916772$
"res_block_output_0/PartitionedCall╔
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall+res_block_output_0/PartitionedCall:output:0conv2d_2_1592375conv2d_2_1592377*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_15916902"
 conv2d_2/StatefulPartitionedCallК
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_1592380conv2d_3_1592382*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_15917072"
 conv2d_3/StatefulPartitionedCallф
add_1/PartitionedCallPartitionedCall+res_block_output_0/PartitionedCall:output:0)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_15917192
add_1/PartitionedCallў
"res_block_output_1/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *X
fSRQ
O__inference_res_block_output_1_layer_call_and_return_conditional_losses_15917262$
"res_block_output_1/PartitionedCall╔
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall+res_block_output_1/PartitionedCall:output:0conv2d_4_1592387conv2d_4_1592389*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_15917392"
 conv2d_4/StatefulPartitionedCallК
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_1592392conv2d_5_1592394*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_15917562"
 conv2d_5/StatefulPartitionedCallф
add_2/PartitionedCallPartitionedCall+res_block_output_1/PartitionedCall:output:0)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_add_2_layer_call_and_return_conditional_losses_15917682
add_2/PartitionedCallў
"res_block_output_2/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *X
fSRQ
O__inference_res_block_output_2_layer_call_and_return_conditional_losses_15917752$
"res_block_output_2/PartitionedCallМ
"value_conv/StatefulPartitionedCallStatefulPartitionedCall+res_block_output_2/PartitionedCall:output:0value_conv_1592399value_conv_1592401*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *P
fKRI
G__inference_value_conv_layer_call_and_return_conditional_losses_15917882$
"value_conv/StatefulPartitionedCall╔
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall+res_block_output_2/PartitionedCall:output:0conv2d_6_1592404conv2d_6_1592406*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_15918052"
 conv2d_6/StatefulPartitionedCallЃ
flatten_1/PartitionedCallPartitionedCall+value_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_15918172
flatten_1/PartitionedCallК
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_1592410conv2d_7_1592412*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_15918292"
 conv2d_7/StatefulPartitionedCallЕ
dense/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1592415dense_1592417*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_15918462
dense/StatefulPartitionedCallч
flatten/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ▄* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_15918582
flatten/PartitionedCallл
$value_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0value_output_1592421value_output_1592423*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *R
fMRK
I__inference_value_output_layer_call_and_return_conditional_losses_15918712&
$value_output/StatefulPartitionedCallё
policy_output/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ▄* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__inference_policy_output_layer_call_and_return_conditional_losses_15918822
policy_output/PartitionedCallѓ
IdentityIdentity&policy_output/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ▄2

Identityї

Identity_1Identity-value_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity_1ђ
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall^dense/StatefulPartitionedCall.^res_block_output_base/StatefulPartitionedCall#^value_conv/StatefulPartitionedCall%^value_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:         : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2^
-res_block_output_base/StatefulPartitionedCall-res_block_output_base/StatefulPartitionedCall2H
"value_conv/StatefulPartitionedCall"value_conv/StatefulPartitionedCall2L
$value_output/StatefulPartitionedCall$value_output/StatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_1
ђ
Џ
.__inference_value_output_layer_call_fn_1593200

inputs
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *R
fMRK
I__inference_value_output_layer_call_and_return_conditional_losses_15918712
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
М
н
'__inference_model_layer_call_fn_1592676

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@#
	unknown_7:@@
	unknown_8:@#
	unknown_9:@@

unknown_10:@$

unknown_11:@@

unknown_12:@$

unknown_13:@

unknown_14:$

unknown_15:@@

unknown_16:@$

unknown_17:@?

unknown_18:?

unknown_19:	└ 

unknown_20: 

unknown_21: 

unknown_22:
identity

identity_1ѕбStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *;
_output_shapes)
':         ▄:         *:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_15922472
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ▄2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:         : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Э
І
R__inference_res_block_output_base_layer_call_and_return_conditional_losses_1592894

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Пb
└
B__inference_model_layer_call_and_return_conditional_losses_1592247

inputs7
res_block_output_base_1592176:@+
res_block_output_base_1592178:@(
conv2d_1592181:@@
conv2d_1592183:@*
conv2d_1_1592186:@@
conv2d_1_1592188:@*
conv2d_2_1592193:@@
conv2d_2_1592195:@*
conv2d_3_1592198:@@
conv2d_3_1592200:@*
conv2d_4_1592205:@@
conv2d_4_1592207:@*
conv2d_5_1592210:@@
conv2d_5_1592212:@,
value_conv_1592217:@ 
value_conv_1592219:*
conv2d_6_1592222:@@
conv2d_6_1592224:@*
conv2d_7_1592228:@?
conv2d_7_1592230:? 
dense_1592233:	└ 
dense_1592235: &
value_output_1592239: "
value_output_1592241:
identity

identity_1ѕбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallб conv2d_2/StatefulPartitionedCallб conv2d_3/StatefulPartitionedCallб conv2d_4/StatefulPartitionedCallб conv2d_5/StatefulPartitionedCallб conv2d_6/StatefulPartitionedCallб conv2d_7/StatefulPartitionedCallбdense/StatefulPartitionedCallб-res_block_output_base/StatefulPartitionedCallб"value_conv/StatefulPartitionedCallб$value_output/StatefulPartitionedCallт
-res_block_output_base/StatefulPartitionedCallStatefulPartitionedCallinputsres_block_output_base_1592176res_block_output_base_1592178*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *[
fVRT
R__inference_res_block_output_base_layer_call_and_return_conditional_losses_15916242/
-res_block_output_base/StatefulPartitionedCall╩
conv2d/StatefulPartitionedCallStatefulPartitionedCall6res_block_output_base/StatefulPartitionedCall:output:0conv2d_1592181conv2d_1592183*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_15916412 
conv2d/StatefulPartitionedCall┼
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_1592186conv2d_1_1592188*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_15916582"
 conv2d_1/StatefulPartitionedCall»
add/PartitionedCallPartitionedCall6res_block_output_base/StatefulPartitionedCall:output:0)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_15916702
add/PartitionedCallќ
"res_block_output_0/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *X
fSRQ
O__inference_res_block_output_0_layer_call_and_return_conditional_losses_15916772$
"res_block_output_0/PartitionedCall╔
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall+res_block_output_0/PartitionedCall:output:0conv2d_2_1592193conv2d_2_1592195*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_15916902"
 conv2d_2/StatefulPartitionedCallК
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_1592198conv2d_3_1592200*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_15917072"
 conv2d_3/StatefulPartitionedCallф
add_1/PartitionedCallPartitionedCall+res_block_output_0/PartitionedCall:output:0)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_15917192
add_1/PartitionedCallў
"res_block_output_1/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *X
fSRQ
O__inference_res_block_output_1_layer_call_and_return_conditional_losses_15917262$
"res_block_output_1/PartitionedCall╔
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall+res_block_output_1/PartitionedCall:output:0conv2d_4_1592205conv2d_4_1592207*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_15917392"
 conv2d_4/StatefulPartitionedCallК
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_1592210conv2d_5_1592212*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_15917562"
 conv2d_5/StatefulPartitionedCallф
add_2/PartitionedCallPartitionedCall+res_block_output_1/PartitionedCall:output:0)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_add_2_layer_call_and_return_conditional_losses_15917682
add_2/PartitionedCallў
"res_block_output_2/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *X
fSRQ
O__inference_res_block_output_2_layer_call_and_return_conditional_losses_15917752$
"res_block_output_2/PartitionedCallМ
"value_conv/StatefulPartitionedCallStatefulPartitionedCall+res_block_output_2/PartitionedCall:output:0value_conv_1592217value_conv_1592219*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *P
fKRI
G__inference_value_conv_layer_call_and_return_conditional_losses_15917882$
"value_conv/StatefulPartitionedCall╔
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall+res_block_output_2/PartitionedCall:output:0conv2d_6_1592222conv2d_6_1592224*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_15918052"
 conv2d_6/StatefulPartitionedCallЃ
flatten_1/PartitionedCallPartitionedCall+value_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_15918172
flatten_1/PartitionedCallК
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_1592228conv2d_7_1592230*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_15918292"
 conv2d_7/StatefulPartitionedCallЕ
dense/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1592233dense_1592235*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_15918462
dense/StatefulPartitionedCallч
flatten/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ▄* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_15918582
flatten/PartitionedCallл
$value_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0value_output_1592239value_output_1592241*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *R
fMRK
I__inference_value_output_layer_call_and_return_conditional_losses_15918712&
$value_output/StatefulPartitionedCallё
policy_output/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ▄* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__inference_policy_output_layer_call_and_return_conditional_losses_15918822
policy_output/PartitionedCallѓ
IdentityIdentity&policy_output/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ▄2

Identityї

Identity_1Identity-value_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity_1ђ
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall^dense/StatefulPartitionedCall.^res_block_output_base/StatefulPartitionedCall#^value_conv/StatefulPartitionedCall%^value_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:         : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2^
-res_block_output_base/StatefulPartitionedCall-res_block_output_base/StatefulPartitionedCall2H
"value_conv/StatefulPartitionedCall"value_conv/StatefulPartitionedCall2L
$value_output/StatefulPartitionedCall$value_output/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
┤
М
%__inference_signature_wrapper_1592566
input_1!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@#
	unknown_7:@@
	unknown_8:@#
	unknown_9:@@

unknown_10:@$

unknown_11:@@

unknown_12:@$

unknown_13:@

unknown_14:$

unknown_15:@@

unknown_16:@$

unknown_17:@?

unknown_18:?

unknown_19:	└ 

unknown_20: 

unknown_21: 

unknown_22:
identity

identity_1ѕбStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *;
_output_shapes)
':         ▄:         *:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *+
f&R$
"__inference__wrapped_model_15916062
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ▄2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:         : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_1
Щ
l
B__inference_add_2_layer_call_and_return_conditional_losses_1591768

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:         @2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         @:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs:WS
/
_output_shapes
:         @
 
_user_specified_nameinputs"еL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ч
serving_defaultу
C
input_18
serving_default_input_1:0         B
policy_output1
StatefulPartitionedCall:0         ▄@
value_output0
StatefulPartitionedCall:1         tensorflow/serving/predict:«Л
н
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer-12
layer-13
layer_with_weights-7
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
layer-17
layer-18
layer_with_weights-10
layer-19
layer-20
layer_with_weights-11
layer-21
	optimizer
loss
trainable_variables
regularization_losses
	variables
	keras_api

signatures
╝__call__
й_default_save_signature
+Й&call_and_return_all_conditional_losses"
_tf_keras_network
"
_tf_keras_input_layer
й

kernel
bias
 trainable_variables
!regularization_losses
"	variables
#	keras_api
┐__call__
+└&call_and_return_all_conditional_losses"
_tf_keras_layer
й

$kernel
%bias
&trainable_variables
'regularization_losses
(	variables
)	keras_api
┴__call__
+┬&call_and_return_all_conditional_losses"
_tf_keras_layer
й

*kernel
+bias
,trainable_variables
-regularization_losses
.	variables
/	keras_api
├__call__
+─&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
0trainable_variables
1regularization_losses
2	variables
3	keras_api
┼__call__
+к&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
4trainable_variables
5regularization_losses
6	variables
7	keras_api
К__call__
+╚&call_and_return_all_conditional_losses"
_tf_keras_layer
й

8kernel
9bias
:trainable_variables
;regularization_losses
<	variables
=	keras_api
╔__call__
+╩&call_and_return_all_conditional_losses"
_tf_keras_layer
й

>kernel
?bias
@trainable_variables
Aregularization_losses
B	variables
C	keras_api
╦__call__
+╠&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
Dtrainable_variables
Eregularization_losses
F	variables
G	keras_api
═__call__
+╬&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
¤__call__
+л&call_and_return_all_conditional_losses"
_tf_keras_layer
й

Lkernel
Mbias
Ntrainable_variables
Oregularization_losses
P	variables
Q	keras_api
Л__call__
+м&call_and_return_all_conditional_losses"
_tf_keras_layer
й

Rkernel
Sbias
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
М__call__
+н&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
Н__call__
+о&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
\trainable_variables
]regularization_losses
^	variables
_	keras_api
О__call__
+п&call_and_return_all_conditional_losses"
_tf_keras_layer
й

`kernel
abias
btrainable_variables
cregularization_losses
d	variables
e	keras_api
┘__call__
+┌&call_and_return_all_conditional_losses"
_tf_keras_layer
й

fkernel
gbias
htrainable_variables
iregularization_losses
j	variables
k	keras_api
█__call__
+▄&call_and_return_all_conditional_losses"
_tf_keras_layer
й

lkernel
mbias
ntrainable_variables
oregularization_losses
p	variables
q	keras_api
П__call__
+я&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
rtrainable_variables
sregularization_losses
t	variables
u	keras_api
▀__call__
+Я&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
vtrainable_variables
wregularization_losses
x	variables
y	keras_api
р__call__
+Р&call_and_return_all_conditional_losses"
_tf_keras_layer
й

zkernel
{bias
|trainable_variables
}regularization_losses
~	variables
	keras_api
с__call__
+С&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
ђtrainable_variables
Ђregularization_losses
ѓ	variables
Ѓ	keras_api
т__call__
+Т&call_and_return_all_conditional_losses"
_tf_keras_layer
├
ёkernel
	Ёbias
єtrainable_variables
Єregularization_losses
ѕ	variables
Ѕ	keras_api
у__call__
+У&call_and_return_all_conditional_losses"
_tf_keras_layer
╝
	іiter
Іbeta_1
їbeta_2

Їdecay
јlearning_ratemїmЇ$mј%mЈ*mљ+mЉ8mњ9mЊ>mћ?mЋLmќMmЌRmўSmЎ`mџamЏfmюgmЮlmъmmЪzmа{mА	ёmб	ЁmБvцvЦ$vд%vД*vе+vЕ8vф9vФ>vг?vГLv«Mv»Rv░Sv▒`v▓av│fv┤gvхlvХmvиzvИ{v╣	ёv║	Ёv╗"
	optimizer
 "
trackable_dict_wrapper
п
0
1
$2
%3
*4
+5
86
97
>8
?9
L10
M11
R12
S13
`14
a15
f16
g17
l18
m19
z20
{21
ё22
Ё23"
trackable_list_wrapper
 "
trackable_list_wrapper
п
0
1
$2
%3
*4
+5
86
97
>8
?9
L10
M11
R12
S13
`14
a15
f16
g17
l18
m19
z20
{21
ё22
Ё23"
trackable_list_wrapper
М
Јmetrics
љlayers
Љlayer_metrics
trainable_variables
regularization_losses
	variables
 њlayer_regularization_losses
Њnon_trainable_variables
╝__call__
й_default_save_signature
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
-
жserving_default"
signature_map
6:4@2res_block_output_base/kernel
(:&@2res_block_output_base/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
х
ћmetrics
Ћlayers
ќlayer_metrics
 trainable_variables
!regularization_losses
"	variables
 Ќlayer_regularization_losses
ўnon_trainable_variables
┐__call__
+└&call_and_return_all_conditional_losses
'└"call_and_return_conditional_losses"
_generic_user_object
':%@@2conv2d/kernel
:@2conv2d/bias
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
х
Ўmetrics
џlayers
Џlayer_metrics
&trainable_variables
'regularization_losses
(	variables
 юlayer_regularization_losses
Юnon_trainable_variables
┴__call__
+┬&call_and_return_all_conditional_losses
'┬"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv2d_1/kernel
:@2conv2d_1/bias
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
х
ъmetrics
Ъlayers
аlayer_metrics
,trainable_variables
-regularization_losses
.	variables
 Аlayer_regularization_losses
бnon_trainable_variables
├__call__
+─&call_and_return_all_conditional_losses
'─"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Бmetrics
цlayers
Цlayer_metrics
0trainable_variables
1regularization_losses
2	variables
 дlayer_regularization_losses
Дnon_trainable_variables
┼__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
еmetrics
Еlayers
фlayer_metrics
4trainable_variables
5regularization_losses
6	variables
 Фlayer_regularization_losses
гnon_trainable_variables
К__call__
+╚&call_and_return_all_conditional_losses
'╚"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv2d_2/kernel
:@2conv2d_2/bias
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
х
Гmetrics
«layers
»layer_metrics
:trainable_variables
;regularization_losses
<	variables
 ░layer_regularization_losses
▒non_trainable_variables
╔__call__
+╩&call_and_return_all_conditional_losses
'╩"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv2d_3/kernel
:@2conv2d_3/bias
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
х
▓metrics
│layers
┤layer_metrics
@trainable_variables
Aregularization_losses
B	variables
 хlayer_regularization_losses
Хnon_trainable_variables
╦__call__
+╠&call_and_return_all_conditional_losses
'╠"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
иmetrics
Иlayers
╣layer_metrics
Dtrainable_variables
Eregularization_losses
F	variables
 ║layer_regularization_losses
╗non_trainable_variables
═__call__
+╬&call_and_return_all_conditional_losses
'╬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
╝metrics
йlayers
Йlayer_metrics
Htrainable_variables
Iregularization_losses
J	variables
 ┐layer_regularization_losses
└non_trainable_variables
¤__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv2d_4/kernel
:@2conv2d_4/bias
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
х
┴metrics
┬layers
├layer_metrics
Ntrainable_variables
Oregularization_losses
P	variables
 ─layer_regularization_losses
┼non_trainable_variables
Л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv2d_5/kernel
:@2conv2d_5/bias
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
х
кmetrics
Кlayers
╚layer_metrics
Ttrainable_variables
Uregularization_losses
V	variables
 ╔layer_regularization_losses
╩non_trainable_variables
М__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
╦metrics
╠layers
═layer_metrics
Xtrainable_variables
Yregularization_losses
Z	variables
 ╬layer_regularization_losses
¤non_trainable_variables
Н__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
лmetrics
Лlayers
мlayer_metrics
\trainable_variables
]regularization_losses
^	variables
 Мlayer_regularization_losses
нnon_trainable_variables
О__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv2d_6/kernel
:@2conv2d_6/bias
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
х
Нmetrics
оlayers
Оlayer_metrics
btrainable_variables
cregularization_losses
d	variables
 пlayer_regularization_losses
┘non_trainable_variables
┘__call__
+┌&call_and_return_all_conditional_losses
'┌"call_and_return_conditional_losses"
_generic_user_object
+:)@2value_conv/kernel
:2value_conv/bias
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
х
┌metrics
█layers
▄layer_metrics
htrainable_variables
iregularization_losses
j	variables
 Пlayer_regularization_losses
яnon_trainable_variables
█__call__
+▄&call_and_return_all_conditional_losses
'▄"call_and_return_conditional_losses"
_generic_user_object
):'@?2conv2d_7/kernel
:?2conv2d_7/bias
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
х
▀metrics
Яlayers
рlayer_metrics
ntrainable_variables
oregularization_losses
p	variables
 Рlayer_regularization_losses
сnon_trainable_variables
П__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Сmetrics
тlayers
Тlayer_metrics
rtrainable_variables
sregularization_losses
t	variables
 уlayer_regularization_losses
Уnon_trainable_variables
▀__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
жmetrics
Жlayers
вlayer_metrics
vtrainable_variables
wregularization_losses
x	variables
 Вlayer_regularization_losses
ьnon_trainable_variables
р__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
:	└ 2dense/kernel
: 2
dense/bias
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
х
Ьmetrics
№layers
­layer_metrics
|trainable_variables
}regularization_losses
~	variables
 ыlayer_regularization_losses
Ыnon_trainable_variables
с__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
зmetrics
Зlayers
шlayer_metrics
ђtrainable_variables
Ђregularization_losses
ѓ	variables
 Шlayer_regularization_losses
эnon_trainable_variables
т__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
%:# 2value_output/kernel
:2value_output/bias
0
ё0
Ё1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
ё0
Ё1"
trackable_list_wrapper
И
Эmetrics
щlayers
Щlayer_metrics
єtrainable_variables
Єregularization_losses
ѕ	variables
 чlayer_regularization_losses
Чnon_trainable_variables
у__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
8
§0
■1
 2"
trackable_list_wrapper
к
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
R

ђtotal

Ђcount
ѓ	variables
Ѓ	keras_api"
_tf_keras_metric
R

ёtotal

Ёcount
є	variables
Є	keras_api"
_tf_keras_metric
R

ѕtotal

Ѕcount
і	variables
І	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
ђ0
Ђ1"
trackable_list_wrapper
.
ѓ	variables"
_generic_user_object
:  (2total
:  (2count
0
ё0
Ё1"
trackable_list_wrapper
.
є	variables"
_generic_user_object
:  (2total
:  (2count
0
ѕ0
Ѕ1"
trackable_list_wrapper
.
і	variables"
_generic_user_object
;:9@2#Adam/res_block_output_base/kernel/m
-:+@2!Adam/res_block_output_base/bias/m
,:*@@2Adam/conv2d/kernel/m
:@2Adam/conv2d/bias/m
.:,@@2Adam/conv2d_1/kernel/m
 :@2Adam/conv2d_1/bias/m
.:,@@2Adam/conv2d_2/kernel/m
 :@2Adam/conv2d_2/bias/m
.:,@@2Adam/conv2d_3/kernel/m
 :@2Adam/conv2d_3/bias/m
.:,@@2Adam/conv2d_4/kernel/m
 :@2Adam/conv2d_4/bias/m
.:,@@2Adam/conv2d_5/kernel/m
 :@2Adam/conv2d_5/bias/m
.:,@@2Adam/conv2d_6/kernel/m
 :@2Adam/conv2d_6/bias/m
0:.@2Adam/value_conv/kernel/m
": 2Adam/value_conv/bias/m
.:,@?2Adam/conv2d_7/kernel/m
 :?2Adam/conv2d_7/bias/m
$:"	└ 2Adam/dense/kernel/m
: 2Adam/dense/bias/m
*:( 2Adam/value_output/kernel/m
$:"2Adam/value_output/bias/m
;:9@2#Adam/res_block_output_base/kernel/v
-:+@2!Adam/res_block_output_base/bias/v
,:*@@2Adam/conv2d/kernel/v
:@2Adam/conv2d/bias/v
.:,@@2Adam/conv2d_1/kernel/v
 :@2Adam/conv2d_1/bias/v
.:,@@2Adam/conv2d_2/kernel/v
 :@2Adam/conv2d_2/bias/v
.:,@@2Adam/conv2d_3/kernel/v
 :@2Adam/conv2d_3/bias/v
.:,@@2Adam/conv2d_4/kernel/v
 :@2Adam/conv2d_4/bias/v
.:,@@2Adam/conv2d_5/kernel/v
 :@2Adam/conv2d_5/bias/v
.:,@@2Adam/conv2d_6/kernel/v
 :@2Adam/conv2d_6/bias/v
0:.@2Adam/value_conv/kernel/v
": 2Adam/value_conv/bias/v
.:,@?2Adam/conv2d_7/kernel/v
 :?2Adam/conv2d_7/bias/v
$:"	└ 2Adam/dense/kernel/v
: 2Adam/dense/bias/v
*:( 2Adam/value_output/kernel/v
$:"2Adam/value_output/bias/v
Ж2у
'__inference_model_layer_call_fn_1591939
'__inference_model_layer_call_fn_1592621
'__inference_model_layer_call_fn_1592676
'__inference_model_layer_call_fn_1592355└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
═B╩
"__inference__wrapped_model_1591606input_1"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
о2М
B__inference_model_layer_call_and_return_conditional_losses_1592775
B__inference_model_layer_call_and_return_conditional_losses_1592874
B__inference_model_layer_call_and_return_conditional_losses_1592429
B__inference_model_layer_call_and_return_conditional_losses_1592503└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
р2я
7__inference_res_block_output_base_layer_call_fn_1592883б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ч2щ
R__inference_res_block_output_base_layer_call_and_return_conditional_losses_1592894б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_conv2d_layer_call_fn_1592903б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_conv2d_layer_call_and_return_conditional_losses_1592914б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_conv2d_1_layer_call_fn_1592923б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1592934б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
¤2╠
%__inference_add_layer_call_fn_1592940б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ж2у
@__inference_add_layer_call_and_return_conditional_losses_1592946б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
я2█
4__inference_res_block_output_0_layer_call_fn_1592951б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щ2Ш
O__inference_res_block_output_0_layer_call_and_return_conditional_losses_1592956б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_conv2d_2_layer_call_fn_1592965б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1592976б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_conv2d_3_layer_call_fn_1592985б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1592996б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_add_1_layer_call_fn_1593002б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
В2ж
B__inference_add_1_layer_call_and_return_conditional_losses_1593008б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
я2█
4__inference_res_block_output_1_layer_call_fn_1593013б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щ2Ш
O__inference_res_block_output_1_layer_call_and_return_conditional_losses_1593018б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_conv2d_4_layer_call_fn_1593027б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1593038б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_conv2d_5_layer_call_fn_1593047б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1593058б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_add_2_layer_call_fn_1593064б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
В2ж
B__inference_add_2_layer_call_and_return_conditional_losses_1593070б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
я2█
4__inference_res_block_output_2_layer_call_fn_1593075б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щ2Ш
O__inference_res_block_output_2_layer_call_and_return_conditional_losses_1593080б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_conv2d_6_layer_call_fn_1593089б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1593100б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
о2М
,__inference_value_conv_layer_call_fn_1593109б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ы2Ь
G__inference_value_conv_layer_call_and_return_conditional_losses_1593120б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_conv2d_7_layer_call_fn_1593129б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_conv2d_7_layer_call_and_return_conditional_losses_1593139б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Н2м
+__inference_flatten_1_layer_call_fn_1593144б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­2ь
F__inference_flatten_1_layer_call_and_return_conditional_losses_1593150б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_flatten_layer_call_fn_1593155б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_flatten_layer_call_and_return_conditional_losses_1593161б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_dense_layer_call_fn_1593170б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
В2ж
B__inference_dense_layer_call_and_return_conditional_losses_1593181б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Т2с
/__inference_policy_output_layer_call_fn_1593186»
д▓б
FullArgSpec%
argsџ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsб

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ђ2■
J__inference_policy_output_layer_call_and_return_conditional_losses_1593191»
д▓б
FullArgSpec%
argsџ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsб

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
п2Н
.__inference_value_output_layer_call_fn_1593200б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
з2­
I__inference_value_output_layer_call_and_return_conditional_losses_1593211б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╠B╔
%__inference_signature_wrapper_1592566input_1"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 ш
"__inference__wrapped_model_1591606╬$%*+89>?LMRSfg`almz{ёЁ8б5
.б+
)і&
input_1         
ф "vфs
9
policy_output(і%
policy_output         ▄
6
value_output&і#
value_output         Р
B__inference_add_1_layer_call_and_return_conditional_losses_1593008Џjбg
`б]
[џX
*і'
inputs/0         @
*і'
inputs/1         @
ф "-б*
#і 
0         @
џ ║
'__inference_add_1_layer_call_fn_1593002јjбg
`б]
[џX
*і'
inputs/0         @
*і'
inputs/1         @
ф " і         @Р
B__inference_add_2_layer_call_and_return_conditional_losses_1593070Џjбg
`б]
[џX
*і'
inputs/0         @
*і'
inputs/1         @
ф "-б*
#і 
0         @
џ ║
'__inference_add_2_layer_call_fn_1593064јjбg
`б]
[џX
*і'
inputs/0         @
*і'
inputs/1         @
ф " і         @Я
@__inference_add_layer_call_and_return_conditional_losses_1592946Џjбg
`б]
[џX
*і'
inputs/0         @
*і'
inputs/1         @
ф "-б*
#і 
0         @
џ И
%__inference_add_layer_call_fn_1592940јjбg
`б]
[џX
*і'
inputs/0         @
*і'
inputs/1         @
ф " і         @х
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1592934l*+7б4
-б*
(і%
inputs         @
ф "-б*
#і 
0         @
џ Ї
*__inference_conv2d_1_layer_call_fn_1592923_*+7б4
-б*
(і%
inputs         @
ф " і         @х
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1592976l897б4
-б*
(і%
inputs         @
ф "-б*
#і 
0         @
џ Ї
*__inference_conv2d_2_layer_call_fn_1592965_897б4
-б*
(і%
inputs         @
ф " і         @х
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1592996l>?7б4
-б*
(і%
inputs         @
ф "-б*
#і 
0         @
џ Ї
*__inference_conv2d_3_layer_call_fn_1592985_>?7б4
-б*
(і%
inputs         @
ф " і         @х
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1593038lLM7б4
-б*
(і%
inputs         @
ф "-б*
#і 
0         @
џ Ї
*__inference_conv2d_4_layer_call_fn_1593027_LM7б4
-б*
(і%
inputs         @
ф " і         @х
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1593058lRS7б4
-б*
(і%
inputs         @
ф "-б*
#і 
0         @
џ Ї
*__inference_conv2d_5_layer_call_fn_1593047_RS7б4
-б*
(і%
inputs         @
ф " і         @х
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1593100l`a7б4
-б*
(і%
inputs         @
ф "-б*
#і 
0         @
џ Ї
*__inference_conv2d_6_layer_call_fn_1593089_`a7б4
-б*
(і%
inputs         @
ф " і         @х
E__inference_conv2d_7_layer_call_and_return_conditional_losses_1593139llm7б4
-б*
(і%
inputs         @
ф "-б*
#і 
0         ?
џ Ї
*__inference_conv2d_7_layer_call_fn_1593129_lm7б4
-б*
(і%
inputs         @
ф " і         ?│
C__inference_conv2d_layer_call_and_return_conditional_losses_1592914l$%7б4
-б*
(і%
inputs         @
ф "-б*
#і 
0         @
џ І
(__inference_conv2d_layer_call_fn_1592903_$%7б4
-б*
(і%
inputs         @
ф " і         @Б
B__inference_dense_layer_call_and_return_conditional_losses_1593181]z{0б-
&б#
!і
inputs         └
ф "%б"
і
0          
џ {
'__inference_dense_layer_call_fn_1593170Pz{0б-
&б#
!і
inputs         └
ф "і          Ф
F__inference_flatten_1_layer_call_and_return_conditional_losses_1593150a7б4
-б*
(і%
inputs         
ф "&б#
і
0         └
џ Ѓ
+__inference_flatten_1_layer_call_fn_1593144T7б4
-б*
(і%
inputs         
ф "і         └Е
D__inference_flatten_layer_call_and_return_conditional_losses_1593161a7б4
-б*
(і%
inputs         ?
ф "&б#
і
0         ▄
џ Ђ
)__inference_flatten_layer_call_fn_1593155T7б4
-б*
(і%
inputs         ?
ф "і         ▄з
B__inference_model_layer_call_and_return_conditional_losses_1592429г$%*+89>?LMRSfg`almz{ёЁ@б=
6б3
)і&
input_1         
p 

 
ф "LбI
Bџ?
і
0/0         ▄
і
0/1         
џ з
B__inference_model_layer_call_and_return_conditional_losses_1592503г$%*+89>?LMRSfg`almz{ёЁ@б=
6б3
)і&
input_1         
p

 
ф "LбI
Bџ?
і
0/0         ▄
і
0/1         
џ Ы
B__inference_model_layer_call_and_return_conditional_losses_1592775Ф$%*+89>?LMRSfg`almz{ёЁ?б<
5б2
(і%
inputs         
p 

 
ф "LбI
Bџ?
і
0/0         ▄
і
0/1         
џ Ы
B__inference_model_layer_call_and_return_conditional_losses_1592874Ф$%*+89>?LMRSfg`almz{ёЁ?б<
5б2
(і%
inputs         
p

 
ф "LбI
Bџ?
і
0/0         ▄
і
0/1         
џ ╩
'__inference_model_layer_call_fn_1591939ъ$%*+89>?LMRSfg`almz{ёЁ@б=
6б3
)і&
input_1         
p 

 
ф ">џ;
і
0         ▄
і
1         ╩
'__inference_model_layer_call_fn_1592355ъ$%*+89>?LMRSfg`almz{ёЁ@б=
6б3
)і&
input_1         
p

 
ф ">џ;
і
0         ▄
і
1         ╔
'__inference_model_layer_call_fn_1592621Ю$%*+89>?LMRSfg`almz{ёЁ?б<
5б2
(і%
inputs         
p 

 
ф ">џ;
і
0         ▄
і
1         ╔
'__inference_model_layer_call_fn_1592676Ю$%*+89>?LMRSfg`almz{ёЁ?б<
5б2
(і%
inputs         
p

 
ф ">џ;
і
0         ▄
і
1         г
J__inference_policy_output_layer_call_and_return_conditional_losses_1593191^4б1
*б'
!і
inputs         ▄

 
ф "&б#
і
0         ▄
џ ё
/__inference_policy_output_layer_call_fn_1593186Q4б1
*б'
!і
inputs         ▄

 
ф "і         ▄╗
O__inference_res_block_output_0_layer_call_and_return_conditional_losses_1592956h7б4
-б*
(і%
inputs         @
ф "-б*
#і 
0         @
џ Њ
4__inference_res_block_output_0_layer_call_fn_1592951[7б4
-б*
(і%
inputs         @
ф " і         @╗
O__inference_res_block_output_1_layer_call_and_return_conditional_losses_1593018h7б4
-б*
(і%
inputs         @
ф "-б*
#і 
0         @
џ Њ
4__inference_res_block_output_1_layer_call_fn_1593013[7б4
-б*
(і%
inputs         @
ф " і         @╗
O__inference_res_block_output_2_layer_call_and_return_conditional_losses_1593080h7б4
-б*
(і%
inputs         @
ф "-б*
#і 
0         @
џ Њ
4__inference_res_block_output_2_layer_call_fn_1593075[7б4
-б*
(і%
inputs         @
ф " і         @┬
R__inference_res_block_output_base_layer_call_and_return_conditional_losses_1592894l7б4
-б*
(і%
inputs         
ф "-б*
#і 
0         @
џ џ
7__inference_res_block_output_base_layer_call_fn_1592883_7б4
-б*
(і%
inputs         
ф " і         @Ѓ
%__inference_signature_wrapper_1592566┘$%*+89>?LMRSfg`almz{ёЁCб@
б 
9ф6
4
input_1)і&
input_1         "vфs
9
policy_output(і%
policy_output         ▄
6
value_output&і#
value_output         и
G__inference_value_conv_layer_call_and_return_conditional_losses_1593120lfg7б4
-б*
(і%
inputs         @
ф "-б*
#і 
0         
џ Ј
,__inference_value_conv_layer_call_fn_1593109_fg7б4
-б*
(і%
inputs         @
ф " і         Ф
I__inference_value_output_layer_call_and_return_conditional_losses_1593211^ёЁ/б,
%б"
 і
inputs          
ф "%б"
і
0         
џ Ѓ
.__inference_value_output_layer_call_fn_1593200QёЁ/б,
%б"
 і
inputs          
ф "і         