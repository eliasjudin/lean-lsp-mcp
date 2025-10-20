namespace EvalPlayground
namespace Basics

def double (n : Nat) : Nat :=
  n + n

def triple (n : Nat) : Nat :=
  double n + n

def offset (n : Nat) : Nat :=
  triple n + 3

def cascade (a b : Nat) : Nat :=
  offset a + double b

end Basics
end EvalPlayground
