class Solution {
public:
    bool isValid(string s) {
        
        std::stack<char> st;;

        for (char c : s) {

            if (!st.empty()) {
                char curr = st.top();
                if (c == ')' && curr == '(') st.pop();
                else if (c == '}' && curr == '{') st.pop();
                else if (c == ']' && curr == '[') st.pop();
                else st.push(c);
            }

            else {
                st.push(c);
            }
        }
        
        return st.empty();
    }
};