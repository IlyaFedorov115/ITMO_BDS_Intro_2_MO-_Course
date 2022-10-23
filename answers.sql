-- 1
SCARD user_list

-- 2
SINTER friends:Elena friends:Boris

-- 3
ZRANK user_r_set Sergey

-- 4
10,9,8,2,3

HMSET basket:104 banana 3 apple 2 carrot 4 potato 2 tomato 3 cucumber 4
HKEYS basket:104
HVALS basket:104
EXPIRE basket:104 300
TTL basket:104

MongoDb
-- 1
db.student.find({surname:"Smith"}).count() 

-- 2
