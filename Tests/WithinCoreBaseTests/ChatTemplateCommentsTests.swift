import Testing

@testable import WithinCoreBase

@Test func trimmedCommentsTakeTheirWhitespace() {
    // The Gemma 4 head: `{{- bos_token -}}`, a trimmed comment, then the first turn.
    let template = "{{- bos_token -}}\n{#- Handle System/Tool Definitions Block -#}\n{%- if x -%}"
    #expect(ChatTemplateComments.strip(template) == "{{- bos_token -}}{%- if x -%}")
    #expect(ChatTemplateComments.strip("a  \n {#- left only #}\nb") == "ab")
    #expect(ChatTemplateComments.strip("a {# right only -#}  \n  b") == "a b")
}

@Test func plainCommentsFollowTransformersBlockTrimming() {
    // lstrip_blocks removes the comment line's indentation; trim_blocks one newline after it.
    #expect(ChatTemplateComments.strip("a\n    {# note #}\nb") == "a\nb")
    #expect(ChatTemplateComments.strip("a {# note #}\n\nb") == "a \nb")
    #expect(ChatTemplateComments.strip("{# first #}x{# second #}") == "x")
}

@Test func templatesWithoutOrWithBrokenCommentsAreUntouched() {
    #expect(ChatTemplateComments.strip("{{ a }}\n{% if b %}c{% endif %}") == "{{ a }}\n{% if b %}c{% endif %}")
    #expect(ChatTemplateComments.strip("a {# never closed") == "a {# never closed")
}
