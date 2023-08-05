#              PANDAS,DATAFRAME TYPES
'''
 * It is an open source library used in data 
   analysis
 * It helps to access different type of data
   format
 * You can be easily slicing big data

 DATAFRAMES

 * It is an collection row and column data
   that was stored as various data types

 * Dataframe can be created in 3 ways

   1. List

   EX: 
     import pandas
     data=[['raju',23],['raja',21],['rajesh',22]]
     DF=pandas.DataFrame(data,index=['a','b','c'],columns=['Name',"Age"])
     print(DF)

   2. Dict

      When we pass dict as an data the key name
      will be changed to column name

      EX: 

      
       import pandas
       data={'name':['raju','ragi','rajan'],'age':[22,23,23]}
       DF=pandas.DataFrame(data,index=['a','b','c'])
       print(DF)

        
   3. Series
    
    * A collection same data type is called series
    * It has dtype argument to specify data type

    EX: 

      import pandas
      data={'one':pandas.Series(['raju','ragi','rajan'],index=[1,2,3]),'two':pandas.Series(['rgu','ranjith','rahul'])}
      DF=pandas.DataFrame(data) 
      print(DF)
 
    * When you declare a dict as series the key
      values are changed as index.

      EX:   

       import pandas as p
       D={'a':100,'b':200,'c':300}
       DF=p.Series(data=D)
       print(DF)

    * We can declare data as list

      EX: import pandas
          D=['raju','chari']
          DF=pandas.Series(D)
          print(DF)
'''

#               SERIES METHODS
'''
 1. axes - returns index from start-end

    import pandas
    D=pandas.Series([100,200,300])
    print(D.axes)

 2. dtype - returns series dataype
 3. empty - returns bool
 4. ndin - returns no of diamensnal
 5. size - returns no of element
 6. values - returns list of items into series
            without index
 7. head(3) - returns specified number of items
             from top.
 8. tail(3) - returns specified number of items
             from bottom.
''' 

#          DATAFRAME FUNCTIONS
'''
 1. axes- returns index
 2  dtype - returns datype
 3. empty - returns bool
 4. ndim - diamansnal
 5. shape - returns number of row and column
 6. size  
 7. values  
 8. head() - returns first 5 rows 
 9. tail() - returns last 5 rows
 10 T      - changes row wise data to column
              column wisw data to row
 EX: 

  import pandas
  D={'name':pandas.Series(['raju','kunaal','gokul']),'Age':pandas.Series([20,20,20])}
  DF=pandas.DataFrame(D,index=range(1,4))
  print(DF.T)
''' 

#           APPLY METHOD IN DF PANDAS
'''
 Apply method helps to apply functions to DF

 EX: 

    import pandas as p
    import numpy as n
    Data=n.random.randint(1,6,[3,3])
    DF=p.DataFrame(Data,columns=['col1','col2','col3'],index=['a','b','c'])
    def double_data(d):
          return d*2
    All=DF.apply(double_data) # it apply all side
    Col=DF['col2'].apply(double_data)
    Row=DF.loc[2:2,2].apply(double_data)
    #print('ALL',DF,All)
    #print(DF,Col)# applyies column side
     print(DF,Row)
'''

#      ITERATION IN SERIES AND DF PANDAS
'''
 When we iteratre series it returns items
 of the series

 import pandas
 Data=[10,20,30]
 Series=pandas.Series(Data)
 for i in Series:
   print(i)

 DF ITERATION
  
  iteritems()-It iterates the value by column return tuple
  iterrow()- It iterates data row wise returns tuple
  itertuples()- It returns row wise data with column name

  import pandas
 Data={"Growth":[1,2],"Items":['orange','Apples']}
 Series=pandas.DataFrame(data=Data)
 print(Series)

 for i in Series.iteritems():
    print(i)

 for i in Series.iterrows():
    print(i)

 for i in Series.itertuples():
   print(i)
'''

#            BASIC STATICS FUNCTION
'''
 SYNTAX: DF_name.count()

 1. count()- returns no of element in each column
 2. sum() - returns sum of each column
 3. mean() - returns average of each column
 4. median() - returns medium value
 5. mode() - returns repated data in each column
 6. min(),max() - returns min,max data
 7. abs() - returns positive values
 8. describe() - returns all of the above in
                 DF format
'''

#         TO_CSV FUNCTION
'''
Helps to change the given data frame into csv file

import pandas as pd
import string
obj=string.ascii_uppercase
data={"Name":["a","b","c"],"Age":["12","20","6",]}
df=pd.DataFrame(data,index=[obj[i] for i in range(len(data["Name"]))])
df.to_csv("data.csv")
print(df)

'''


#    DF INDEXING
'''
 Dfname["ColumnName"][index_value]=value
'''

#   DIFFERENCE BETWEEN DF AND SERIED
'''
* Series is an column in a table, contains 1
  dimensional array data, that may contain different
  types of data, it contians either only row or
  column

* DF is 2(rows,column) dimensional labeled data with different
  data types.
'''
