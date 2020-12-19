(ns clojure-noob.junk
  (:require [my.lib :as bar]))

(def m #::bar{:kw 1, :n/kw 2, :_/bare 3, 0 4})
(+ (bar/a 1) (b 2))

;; (dissoc (assoc () :key \"value \") :lock)

'("a" "b" :c)

(defn abcd [a b] a)
(defn f [b] b)

(defn tata [n] (str n "ta"))
(tata 3)
(defn ta [n] n)
(ta 5)

(def x 15)

(defn foobar "fooooobaaaar" [f b] (str f b))

(foobar "abc" "def")

(defn add [a b]
  (+ a b))

(if this
  (if that
    (then AAA)
    (else BBB))
  (otherwise CCC))

{:status 200
 :body (let [body (find-body abc)]
         body)}

(defn handle-request
  (let [body (find-body abc)
        status (or status 500)]
    {:status status
     :body body}))

(defn handle-request []
  (let [body (find-body abc)]
    {:status 200
     :length (count body)
     :body body})
  (println (find-body abc))
  (println \"foobar \"))
