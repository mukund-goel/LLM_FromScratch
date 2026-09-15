# Comparison of Attention Mechanisms in Tiny GPT

## 1. Overview

This project implements and compares three attention mechanisms in the same small character-level GPT model: dense causal attention, sliding-window causal attention, and BigBird-style sparse attention using local, random, and global connections. The purpose was to study how restricting the attention pattern affects the information available to the model and whether the reduction in computation comes at a significant cost in performance.

The model was trained on the Tiny Shakespeare dataset, which contains approximately 1.1 million characters with a vocabulary of 65 characters. The model uses an embedding dimension of 64, a Q/K dimension of 32, a V dimension of 64, a context length of 128, a batch size of 32, and two Transformer layers. Each layer consists of Q/K/V projections, attention, a residual connection, a feed-forward network, and another residual connection.

## 2. Attention Implementations

Dense attention was used as the reference implementation. Each token can attend to every previous token as well as itself, while future tokens are masked. This gives the model access to the complete previous context. The main drawback is the quadratic growth in the number of attention interactions with sequence length.

Sliding-window attention restricts each query to a fixed number of recent tokens. In this implementation, the window size was 8. The valid range is constructed using the current position and the window size, so a query only attends to its local history and itself. This significantly reduces the number of attention interactions, but removes direct access to information that lies outside the window.

BigBird-style attention extends this idea by combining local, random, and global connections. Local connections provide nearby context, random connections provide additional long-range paths, and global connections allow selected tokens to communicate information across distant parts of the sequence. This gives the model more long-range information than a purely sliding-window pattern without using the complete dense attention pattern.

## 3. What Information Is Lost?

The main information lost when moving from dense attention to sparse attention is direct access to distant context. With dense attention, a token can directly use information from any previous position. With sliding-window attention, information outside the window is not directly available to the current token.

This does not mean that distant information is completely lost. It can still be passed indirectly through intermediate token representations. However, this requires information to propagate through multiple positions. Since the model used here has only two Transformer layers, there are limited opportunities for this indirect propagation.

BigBird-style attention addresses this by adding random and global connections. These connections create shortcuts between distant parts of the sequence and allow information to travel without depending entirely on local propagation.

## 4. Importance of Global Tokens

Global tokens matter because they provide a way of carrying long-term information and context across the sequence. A piece of information that occurred earlier may no longer be inside the sliding window when the model reaches the position where it becomes useful.

A global connection allows this information to remain accessible through a long-range path. This can be useful when generating the next token because the prediction may depend on something that happened earlier in the sequence but is no longer covered by the local window.

This is particularly useful in this model because it only has two layers. With more layers, local information could potentially propagate across larger distances through repeated attention operations. Here, global connections provide a more direct way of preserving long-range context.

## 5. NaN Handling

A possible issue in sparse attention occurs when the entire allowed set for a query is masked. In that situation, there are no valid positions from which the query can obtain information, which can cause invalid values during the softmax operation.

This is a real issue that can occur when causal and sparse masks are combined incorrectly, particularly around sequence boundaries.

In the attention mechanisms implemented in this project, this case cannot occur because the current token is always included in its own allowed attention set. Therefore, every query has at least one valid key: itself.

For a query at position \(i\), position \(i\) is always available. This remains true even at the beginning of the sequence. As a result, there is never a query where all attention positions are masked.

Thus, NaN handling is effectively built into the attention-mask design itself. There is no separate post-softmax NaN correction required because the implementation guarantees that every query has at least one valid attention position.

## 6. Experimental Results

The three models performed nearly equally overall. The dense attention model performed the best, while the sliding-window and BigBird-style models were close behind and only slightly worse.

The dense model capturing the most complete context makes it the ideal reference case. It does not have to rely on indirect information propagation or sparse long-range connections. Every previous token is directly available to every query.

At the same time, the performance difference between the models was relatively small. The sparse models still performed very well despite using considerably fewer attention connections. This is important because the reduction in attention connections also resulted in substantially less processing time.

The trade-off can therefore be summarized simply: dense attention gives the model the most complete contextual information and the best performance, while sparse attention gives up some direct contextual access in exchange for significantly lower processing requirements.

## 7. Conclusion

The experiment shows that dense attention remains the ideal case when the goal is to give the model complete access to its previous context. It achieved the best performance among the three implementations.

However, the sparse mechanisms performed surprisingly close to dense attention. Sliding-window attention mainly loses direct access to distant context, while BigBird-style attention recovers part of this through random and global connections. Global tokens are especially useful for carrying long-term information that falls outside the local window and can therefore help with later next-token predictions.

The most useful result of the comparison is the trade-off between performance and computation. The sparse models were only slightly worse than dense attention while requiring substantially less processing time. This suggests that carefully chosen sparse attention patterns can retain most of the useful contextual information without requiring every possible query-key interaction.

Overall, dense attention provides the strongest reference performance, while sliding-window and BigBird-style attention demonstrate that much of this performance can be retained with a significantly cheaper attention pattern.
