-module(xxx).
-export([start/0]).
-import(data, [get_X/0]).

% взять К-ый столбец  (у вас уже есть)
getKth(_,[],Akk) ->
    lists:reverse(Akk);
getKth(K,[H|T],Akk) ->
    getKth(K,T,[lists:nth(K,H)|Akk]).
%----------------------------------------

start()->  X = get_X(),
		   X1i=getKth(1,X,[]),
		   X2i=getKth(2,X,[]),
		   
           Max1=lists:max(X1i)+0.05,
		   Min1=lists:min(X1i)-0.05,
		   Max2=lists:max(X2i)+0.05,
		   Min2=lists:min(X2i)-0.05,
		   generate(Min1,Min2,Max1,Max2,Min2,[]).

 generate(A,B,C,D,Min2,Akk) ->  if A>C -> Akk;
                                   A=<C ->  
                                        if B=<D -> generate(A,B+0.01,C,D,Min2,[[A,B]|Akk]);
								            B>D -> generate(A+0.01,Min2,C,D,Min2,[[A,B]|Akk])
								        end
                                   end.